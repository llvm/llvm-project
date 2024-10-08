//==- WebAssemblyAsmTypeCheck.cpp - Assembler for WebAssembly -*- C++ -*-==//
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

#include "AsmParser/WebAssemblyAsmTypeCheck.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "MCTargetDesc/WebAssemblyMCTypeUtilities.h"
#include "MCTargetDesc/WebAssemblyTargetStreamer.h"
#include "TargetInfo/WebAssemblyTargetInfo.h"
#include "WebAssembly.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SourceMgr.h"
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "wasm-asm-parser"

extern StringRef getMnemonic(unsigned Opc);

namespace llvm {

WebAssemblyAsmTypeCheck::WebAssemblyAsmTypeCheck(MCAsmParser &Parser,
                                                 const MCInstrInfo &MII,
                                                 bool Is64)
    : Parser(Parser), MII(MII), Is64(Is64) {}

void WebAssemblyAsmTypeCheck::funcDecl(const wasm::WasmSignature &Sig) {
  LocalTypes.assign(Sig.Params.begin(), Sig.Params.end());
  ReturnTypes.assign(Sig.Returns.begin(), Sig.Returns.end());
  BrStack.emplace_back(Sig.Returns.begin(), Sig.Returns.end());
}

void WebAssemblyAsmTypeCheck::localDecl(
    const SmallVectorImpl<wasm::ValType> &Locals) {
  LocalTypes.insert(LocalTypes.end(), Locals.begin(), Locals.end());
}

void WebAssemblyAsmTypeCheck::dumpTypeStack(Twine Msg) {
  LLVM_DEBUG({ dbgs() << Msg << getTypesString(Stack, 0) << "\n"; });
}

bool WebAssemblyAsmTypeCheck::typeError(SMLoc ErrorLoc, const Twine &Msg) {
  // If we're currently in unreachable code, we suppress errors completely.
  if (Unreachable)
    return false;
  dumpTypeStack("current stack: ");
  return Parser.Error(ErrorLoc, Msg);
}

bool WebAssemblyAsmTypeCheck::match(StackType TypeA, StackType TypeB) {
  if (TypeA == TypeB)
    return false;
  if (std::get_if<Any>(&TypeA) || std::get_if<Any>(&TypeB))
    return false;

  if (std::get_if<Ref>(&TypeB))
    std::swap(TypeA, TypeB);
  assert(std::get_if<wasm::ValType>(&TypeB));
  if (std::get_if<Ref>(&TypeA) &&
      WebAssembly::isRefType(std::get<wasm::ValType>(TypeB)))
    return false;
  return true;
}

std::string WebAssemblyAsmTypeCheck::getTypesString(ArrayRef<StackType> Types,
                                                    size_t StartPos) {
  SmallVector<std::string, 4> Reverse;
  for (auto I = Types.size(); I > StartPos; I--) {
    if (std::get_if<Any>(&Types[I - 1]))
      Reverse.push_back("any");
    else if (std::get_if<Ref>(&Types[I - 1]))
      Reverse.push_back("ref");
    else
      Reverse.push_back(
          WebAssembly::typeToString(std::get<wasm::ValType>(Types[I - 1])));
  }

  std::stringstream SS;
  SS << "[";
  bool First = true;
  for (auto It = Reverse.rbegin(); It != Reverse.rend(); ++It) {
    if (!First)
      SS << ", ";
    SS << *It;
    First = false;
  }
  SS << "]";
  return SS.str();
}

SmallVector<WebAssemblyAsmTypeCheck::StackType, 4>
WebAssemblyAsmTypeCheck::valTypeToStackType(ArrayRef<wasm::ValType> ValTypes) {
  SmallVector<StackType, 4> Types(ValTypes.size());
  std::transform(ValTypes.begin(), ValTypes.end(), Types.begin(),
                 [](wasm::ValType Val) -> StackType { return Val; });
  return Types;
}

bool WebAssemblyAsmTypeCheck::checkTypes(SMLoc ErrorLoc,
                                         ArrayRef<wasm::ValType> ValTypes,
                                         bool ExactMatch) {
  return checkTypes(ErrorLoc, valTypeToStackType(ValTypes), ExactMatch);
}

bool WebAssemblyAsmTypeCheck::checkTypes(SMLoc ErrorLoc,
                                         ArrayRef<StackType> Types,
                                         bool ExactMatch) {
  auto StackI = Stack.size();
  auto TypeI = Types.size();
  bool Error = false;
  // Compare elements one by one from the stack top
  for (; StackI > 0 && TypeI > 0; StackI--, TypeI--) {
    if (match(Stack[StackI - 1], Types[TypeI - 1])) {
      Error = true;
      break;
    }
  }
  // Even if no match failure has happened in the loop above, if not all
  // elements of Types has been matched, that means we don't have enough
  // elements on the stack.
  //
  // Also, if not all elements of the Stack has been matched and when
  // 'ExactMatch' is true, that means we have superfluous elements remaining on
  // the stack (e.g. at the end of a function).
  if (TypeI > 0 || (ExactMatch && StackI > 0))
    Error = true;

  if (!Error)
    return false;

  auto StackStartPos =
      ExactMatch ? 0 : std::max(0, (int)Stack.size() - (int)Types.size());
  return typeError(ErrorLoc, "type mismatch, expected " +
                                 getTypesString(Types, 0) + " but got " +
                                 getTypesString(Stack, StackStartPos));
}

bool WebAssemblyAsmTypeCheck::checkAndPopTypes(SMLoc ErrorLoc,
                                               ArrayRef<wasm::ValType> ValTypes,
                                               bool ExactMatch) {
  return checkAndPopTypes(ErrorLoc, valTypeToStackType(ValTypes), ExactMatch);
}

bool WebAssemblyAsmTypeCheck::checkAndPopTypes(SMLoc ErrorLoc,
                                               ArrayRef<StackType> Types,
                                               bool ExactMatch) {
  bool Error = checkTypes(ErrorLoc, Types, ExactMatch);
  auto NumPops = std::min(Stack.size(), Types.size());
  for (size_t I = 0, E = NumPops; I != E; I++)
    Stack.pop_back();
  return Error;
}

bool WebAssemblyAsmTypeCheck::popType(SMLoc ErrorLoc, StackType Type) {
  return checkAndPopTypes(ErrorLoc, {Type}, false);
}

bool WebAssemblyAsmTypeCheck::popRefType(SMLoc ErrorLoc) {
  return popType(ErrorLoc, Ref{});
}

bool WebAssemblyAsmTypeCheck::popAnyType(SMLoc ErrorLoc) {
  return popType(ErrorLoc, Any{});
}

void WebAssemblyAsmTypeCheck::pushTypes(ArrayRef<wasm::ValType> ValTypes) {
  Stack.append(valTypeToStackType(ValTypes));
}

bool WebAssemblyAsmTypeCheck::getLocal(SMLoc ErrorLoc, const MCOperand &LocalOp,
                                       wasm::ValType &Type) {
  auto Local = static_cast<size_t>(LocalOp.getImm());
  if (Local >= LocalTypes.size())
    return typeError(ErrorLoc, StringRef("no local type specified for index ") +
                                   std::to_string(Local));
  Type = LocalTypes[Local];
  return false;
}

bool WebAssemblyAsmTypeCheck::checkBr(SMLoc ErrorLoc, size_t Level) {
  if (Level >= BrStack.size())
    return typeError(ErrorLoc,
                     StringRef("br: invalid depth ") + std::to_string(Level));
  const SmallVector<wasm::ValType, 4> &Expected =
      BrStack[BrStack.size() - Level - 1];
  return checkTypes(ErrorLoc, Expected, false);
  return false;
}

bool WebAssemblyAsmTypeCheck::checkEnd(SMLoc ErrorLoc, bool PopVals) {
  if (!PopVals)
    BrStack.pop_back();

  if (PopVals)
    return checkAndPopTypes(ErrorLoc, LastSig.Returns, false);
  return checkTypes(ErrorLoc, LastSig.Returns, false);
}

bool WebAssemblyAsmTypeCheck::checkSig(SMLoc ErrorLoc,
                                       const wasm::WasmSignature &Sig) {
  bool Error = checkAndPopTypes(ErrorLoc, Sig.Params, false);
  pushTypes(Sig.Returns);
  return Error;
}

bool WebAssemblyAsmTypeCheck::getSymRef(SMLoc ErrorLoc, const MCOperand &SymOp,
                                        const MCSymbolRefExpr *&SymRef) {
  if (!SymOp.isExpr())
    return typeError(ErrorLoc, StringRef("expected expression operand"));
  SymRef = dyn_cast<MCSymbolRefExpr>(SymOp.getExpr());
  if (!SymRef)
    return typeError(ErrorLoc, StringRef("expected symbol operand"));
  return false;
}

bool WebAssemblyAsmTypeCheck::getGlobal(SMLoc ErrorLoc,
                                        const MCOperand &GlobalOp,
                                        wasm::ValType &Type) {
  const MCSymbolRefExpr *SymRef;
  if (getSymRef(ErrorLoc, GlobalOp, SymRef))
    return true;
  const auto *WasmSym = cast<MCSymbolWasm>(&SymRef->getSymbol());
  switch (WasmSym->getType().value_or(wasm::WASM_SYMBOL_TYPE_DATA)) {
  case wasm::WASM_SYMBOL_TYPE_GLOBAL:
    Type = static_cast<wasm::ValType>(WasmSym->getGlobalType().Type);
    break;
  case wasm::WASM_SYMBOL_TYPE_FUNCTION:
  case wasm::WASM_SYMBOL_TYPE_DATA:
    switch (SymRef->getKind()) {
    case MCSymbolRefExpr::VK_GOT:
    case MCSymbolRefExpr::VK_WASM_GOT_TLS:
      Type = Is64 ? wasm::ValType::I64 : wasm::ValType::I32;
      return false;
    default:
      break;
    }
    [[fallthrough]];
  default:
    return typeError(ErrorLoc, StringRef("symbol ") + WasmSym->getName() +
                                   ": missing .globaltype");
  }
  return false;
}

bool WebAssemblyAsmTypeCheck::getTable(SMLoc ErrorLoc, const MCOperand &TableOp,
                                       wasm::ValType &Type) {
  const MCSymbolRefExpr *SymRef;
  if (getSymRef(ErrorLoc, TableOp, SymRef))
    return true;
  const auto *WasmSym = cast<MCSymbolWasm>(&SymRef->getSymbol());
  if (WasmSym->getType().value_or(wasm::WASM_SYMBOL_TYPE_DATA) !=
      wasm::WASM_SYMBOL_TYPE_TABLE)
    return typeError(ErrorLoc, StringRef("symbol ") + WasmSym->getName() +
                                   ": missing .tabletype");
  Type = static_cast<wasm::ValType>(WasmSym->getTableType().ElemType);
  return false;
}

bool WebAssemblyAsmTypeCheck::getSignature(SMLoc ErrorLoc,
                                           const MCOperand &SigOp,
                                           wasm::WasmSymbolType Type,
                                           const wasm::WasmSignature *&Sig) {
  const MCSymbolRefExpr *SymRef = nullptr;
  if (getSymRef(ErrorLoc, SigOp, SymRef))
    return true;
  const auto *WasmSym = cast<MCSymbolWasm>(&SymRef->getSymbol());
  Sig = WasmSym->getSignature();

  if (!Sig || WasmSym->getType() != Type) {
    const char *TypeName = nullptr;
    switch (Type) {
    case wasm::WASM_SYMBOL_TYPE_FUNCTION:
      TypeName = "func";
      break;
    case wasm::WASM_SYMBOL_TYPE_TAG:
      TypeName = "tag";
      break;
    default:
      llvm_unreachable("Signature symbol should either be a function or a tag");
    }
    return typeError(ErrorLoc, StringRef("symbol ") + WasmSym->getName() +
                                   ": missing ." + TypeName + "type");
  }
  return false;
}

bool WebAssemblyAsmTypeCheck::endOfFunction(SMLoc ErrorLoc, bool ExactMatch) {
  bool Error = checkAndPopTypes(ErrorLoc, ReturnTypes, ExactMatch);
  Unreachable = true;
  return Error;
}

bool WebAssemblyAsmTypeCheck::typeCheck(SMLoc ErrorLoc, const MCInst &Inst,
                                        OperandVector &Operands) {
  auto Opc = Inst.getOpcode();
  auto Name = getMnemonic(Opc);
  dumpTypeStack("typechecking " + Name + ": ");
  wasm::ValType Type;

  if (Name == "local.get") {
    if (!getLocal(Operands[1]->getStartLoc(), Inst.getOperand(0), Type)) {
      pushType(Type);
      return false;
    }
    return true;
  }

  if (Name == "local.set") {
    if (!getLocal(Operands[1]->getStartLoc(), Inst.getOperand(0), Type))
      return popType(ErrorLoc, Type);
    return true;
  }

  if (Name == "local.tee") {
    if (!getLocal(Operands[1]->getStartLoc(), Inst.getOperand(0), Type)) {
      bool Error = popType(ErrorLoc, Type);
      pushType(Type);
      return Error;
    }
    return true;
  }

  if (Name == "global.get") {
    if (!getGlobal(Operands[1]->getStartLoc(), Inst.getOperand(0), Type)) {
      pushType(Type);
      return false;
    }
    return true;
  }

  if (Name == "global.set") {
    if (!getGlobal(Operands[1]->getStartLoc(), Inst.getOperand(0), Type))
      return popType(ErrorLoc, Type);
    return true;
  }

  if (Name == "table.get") {
    bool Error = popType(ErrorLoc, wasm::ValType::I32);
    if (!getTable(Operands[1]->getStartLoc(), Inst.getOperand(0), Type)) {
      pushType(Type);
      return Error;
    }
    return true;
  }

  if (Name == "table.set") {
    bool Error = false;
    if (!getTable(Operands[1]->getStartLoc(), Inst.getOperand(0), Type))
      Error |= popType(ErrorLoc, Type);
    else
      Error = true;
    Error |= popType(ErrorLoc, wasm::ValType::I32);
    return Error;
  }

  if (Name == "table.size") {
    bool Error = getTable(Operands[1]->getStartLoc(), Inst.getOperand(0), Type);
    pushType(wasm::ValType::I32);
    return Error;
  }

  if (Name == "table.grow") {
    bool Error = popType(ErrorLoc, wasm::ValType::I32);
    if (!getTable(Operands[1]->getStartLoc(), Inst.getOperand(0), Type))
      Error |= popType(ErrorLoc, Type);
    else
      Error = true;
    pushType(wasm::ValType::I32);
    return Error;
  }

  if (Name == "table.fill") {
    bool Error = popType(ErrorLoc, wasm::ValType::I32);
    if (!getTable(Operands[1]->getStartLoc(), Inst.getOperand(0), Type))
      Error |= popType(ErrorLoc, Type);
    else
      Error = true;
    Error |= popType(ErrorLoc, wasm::ValType::I32);
    return Error;
  }

  if (Name == "memory.fill") {
    Type = Is64 ? wasm::ValType::I64 : wasm::ValType::I32;
    bool Error = popType(ErrorLoc, Type);
    Error |= popType(ErrorLoc, wasm::ValType::I32);
    Error |= popType(ErrorLoc, Type);
    return Error;
  }

  if (Name == "memory.copy") {
    Type = Is64 ? wasm::ValType::I64 : wasm::ValType::I32;
    bool Error = popType(ErrorLoc, Type);
    Error |= popType(ErrorLoc, Type);
    Error |= popType(ErrorLoc, Type);
    return Error;
  }

  if (Name == "memory.init") {
    Type = Is64 ? wasm::ValType::I64 : wasm::ValType::I32;
    bool Error = popType(ErrorLoc, wasm::ValType::I32);
    Error |= popType(ErrorLoc, wasm::ValType::I32);
    Error |= popType(ErrorLoc, Type);
    return Error;
  }

  if (Name == "drop") {
    return popType(ErrorLoc, Any{});
  }

  if (Name == "try" || Name == "block" || Name == "loop" || Name == "if") {
    if (Name == "loop")
      BrStack.emplace_back(LastSig.Params.begin(), LastSig.Params.end());
    else
      BrStack.emplace_back(LastSig.Returns.begin(), LastSig.Returns.end());
    if (Name == "if" && popType(ErrorLoc, wasm::ValType::I32))
      return true;
    return false;
  }

  if (Name == "end_block" || Name == "end_loop" || Name == "end_if" ||
      Name == "else" || Name == "end_try" || Name == "catch" ||
      Name == "catch_all" || Name == "delegate") {
    bool Error = checkEnd(ErrorLoc, Name == "else" || Name == "catch" ||
                                        Name == "catch_all");
    Unreachable = false;
    if (Name == "catch") {
      const wasm::WasmSignature *Sig = nullptr;
      if (!getSignature(Operands[1]->getStartLoc(), Inst.getOperand(0),
                        wasm::WASM_SYMBOL_TYPE_TAG, Sig))
        // catch instruction pushes values whose types are specified in the
        // tag's "params" part
        pushTypes(Sig->Params);
      else
        Error = true;
    }
    return Error;
  }

  if (Name == "br") {
    const MCOperand &Operand = Inst.getOperand(0);
    if (!Operand.isImm())
      return true;
    return checkBr(ErrorLoc, static_cast<size_t>(Operand.getImm()));
  }

  if (Name == "return") {
    return endOfFunction(ErrorLoc, false);
  }

  if (Name == "call_indirect" || Name == "return_call_indirect") {
    // Function value.
    bool Error = popType(ErrorLoc, wasm::ValType::I32);
    Error |= checkSig(ErrorLoc, LastSig);
    if (Name == "return_call_indirect" && endOfFunction(ErrorLoc, false))
      return true;
    return Error;
  }

  if (Name == "call" || Name == "return_call") {
    bool Error = false;
    const wasm::WasmSignature *Sig = nullptr;
    if (!getSignature(Operands[1]->getStartLoc(), Inst.getOperand(0),
                      wasm::WASM_SYMBOL_TYPE_FUNCTION, Sig))
      Error |= checkSig(ErrorLoc, *Sig);
    else
      Error = true;
    if (Name == "return_call" && endOfFunction(ErrorLoc, false))
      return true;
    return Error;
  }

  if (Name == "unreachable") {
    Unreachable = true;
    return false;
  }

  if (Name == "ref.is_null") {
    bool Error = popRefType(ErrorLoc);
    pushType(wasm::ValType::I32);
    return Error;
  }

  if (Name == "throw") {
    const wasm::WasmSignature *Sig = nullptr;
    if (!getSignature(Operands[1]->getStartLoc(), Inst.getOperand(0),
                      wasm::WASM_SYMBOL_TYPE_TAG, Sig))
      return checkSig(ErrorLoc, *Sig);
    return true;
  }

  // The current instruction is a stack instruction which doesn't have
  // explicit operands that indicate push/pop types, so we get those from
  // the register version of the same instruction.
  auto RegOpc = WebAssembly::getRegisterOpcode(Opc);
  assert(RegOpc != -1 && "Failed to get register version of MC instruction");
  const auto &II = MII.get(RegOpc);
  // First pop all the uses off the stack and check them.
  SmallVector<wasm::ValType, 4> PopTypes;
  for (unsigned I = II.getNumDefs(); I < II.getNumOperands(); I++) {
    const auto &Op = II.operands()[I];
    if (Op.OperandType == MCOI::OPERAND_REGISTER)
      PopTypes.push_back(WebAssembly::regClassToValType(Op.RegClass));
  }
  bool Error = checkAndPopTypes(ErrorLoc, PopTypes, false);
  SmallVector<wasm::ValType, 4> PushTypes;
  // Now push all the defs onto the stack.
  for (unsigned I = 0; I < II.getNumDefs(); I++) {
    const auto &Op = II.operands()[I];
    assert(Op.OperandType == MCOI::OPERAND_REGISTER && "Register expected");
    PushTypes.push_back(WebAssembly::regClassToValType(Op.RegClass));
  }
  pushTypes(PushTypes);
  return Error;
}

} // end namespace llvm
