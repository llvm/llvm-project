//===- WebAssemblyMCTypeUtilities.cpp - WebAssembly Type Utility Functions-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements several utility functions for WebAssembly type parsing.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCTypeUtilities.h"
#include "WebAssemblyMCTargetDesc.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;

std::optional<wasm::ValType> WebAssembly::parseType(StringRef Type) {
  return llvm::StringSwitch<std::optional<wasm::ValType>>{Type}
      .Case("i32", wasm::ValType::I32)
      .Case("i64", wasm::ValType::I64)
      .Case("f32", wasm::ValType::F32)
      .Case("f64", wasm::ValType::F64)
      .Cases({"v128", "i8x16", "i16x8", "i32x4", "i64x2", "f32x4", "f64x2"},
             wasm::ValType::V128)
      .Case("funcref", wasm::ValType::FUNCREF)
      .Case("externref", wasm::ValType::EXTERNREF)
      .Case("exnref", wasm::ValType::EXNREF)
      .Default(std::nullopt);
}

WebAssembly::BlockType WebAssembly::parseBlockType(StringRef Type) {
  // Multivalue block types are handled separately in parseSignature
  return StringSwitch<WebAssembly::BlockType>(Type)
      .Case("i32", WebAssembly::BlockType::I32)
      .Case("i64", WebAssembly::BlockType::I64)
      .Case("f32", WebAssembly::BlockType::F32)
      .Case("f64", WebAssembly::BlockType::F64)
      .Case("v128", WebAssembly::BlockType::V128)
      .Case("funcref", WebAssembly::BlockType::Funcref)
      .Case("externref", WebAssembly::BlockType::Externref)
      .Case("exnref", WebAssembly::BlockType::Exnref)
      .Case("void", WebAssembly::BlockType::Void)
      .Default(WebAssembly::BlockType::Invalid);
}

// We have various enums representing a subset of these types, use this
// function to convert any of them to text.
const char *WebAssembly::anyTypeToString(unsigned Type) {
  switch (Type) {
  case wasm::WASM_TYPE_I32:
    return "i32";
  case wasm::WASM_TYPE_I64:
    return "i64";
  case wasm::WASM_TYPE_F32:
    return "f32";
  case wasm::WASM_TYPE_F64:
    return "f64";
  case wasm::WASM_TYPE_V128:
    return "v128";
  case wasm::WASM_TYPE_FUNCREF:
    return "funcref";
  case wasm::WASM_TYPE_EXTERNREF:
    return "externref";
  case wasm::WASM_TYPE_EXNREF:
    return "exnref";
  case wasm::WASM_TYPE_FUNC:
    return "func";
  case wasm::WASM_TYPE_NORESULT:
    return "void";
  default:
    return "invalid_type";
  }
}

const char *WebAssembly::typeToString(wasm::ValType Type) {
  return anyTypeToString(static_cast<unsigned>(Type));
}

std::string WebAssembly::typeListToString(ArrayRef<wasm::ValType> List) {
  std::string S;
  for (const auto &Type : List) {
    if (&Type != &List[0])
      S += ", ";
    S += WebAssembly::typeToString(Type);
  }
  return S;
}

std::string WebAssembly::signatureToString(const wasm::WasmSignature *Sig) {
  std::string S("(");
  S += typeListToString(Sig->Params);
  S += ") -> (";
  S += typeListToString(Sig->Returns);
  S += ")";
  return S;
}

wasm::ValType WebAssembly::regClassToValType(unsigned RC) {
  switch (RC) {
  case WebAssembly::I32RegClassID:
    return wasm::ValType::I32;
  case WebAssembly::I64RegClassID:
    return wasm::ValType::I64;
  case WebAssembly::F32RegClassID:
    return wasm::ValType::F32;
  case WebAssembly::F64RegClassID:
    return wasm::ValType::F64;
  case WebAssembly::V128RegClassID:
    return wasm::ValType::V128;
  case WebAssembly::FUNCREFRegClassID:
    return wasm::ValType::FUNCREF;
  case WebAssembly::EXTERNREFRegClassID:
    return wasm::ValType::EXTERNREF;
  case WebAssembly::EXNREFRegClassID:
    return wasm::ValType::EXNREF;
  default:
    llvm_unreachable("unexpected type");
  }
}

std::optional<wasm::WasmInitExpr>
WebAssembly::makeGlobalInitExpr(wasm::ValType Type, uint64_t Bits) {
  wasm::WasmInitExpr Expr;
  Expr.Extended = false;
  switch (Type) {
  case wasm::ValType::I32:
    Expr.Inst.Opcode = wasm::WASM_OPCODE_I32_CONST;
    Expr.Inst.Value.Int32 = static_cast<int32_t>(static_cast<uint32_t>(Bits));
    return Expr;
  case wasm::ValType::I64:
    Expr.Inst.Opcode = wasm::WASM_OPCODE_I64_CONST;
    Expr.Inst.Value.Int64 = static_cast<int64_t>(Bits);
    return Expr;
  case wasm::ValType::F32:
    Expr.Inst.Opcode = wasm::WASM_OPCODE_F32_CONST;
    Expr.Inst.Value.Float32 = static_cast<uint32_t>(Bits);
    return Expr;
  case wasm::ValType::F64:
    Expr.Inst.Opcode = wasm::WASM_OPCODE_F64_CONST;
    Expr.Inst.Value.Float64 = Bits;
    return Expr;
  default:
    return std::nullopt;
  }
}

std::string WebAssembly::floatToString(const APFloat &FP) {
  // Print NaNs with custom payloads specially.
  if (FP.isNaN() && !FP.bitwiseIsEqual(APFloat::getQNaN(FP.getSemantics())) &&
      !FP.bitwiseIsEqual(
          APFloat::getQNaN(FP.getSemantics(), /*Negative=*/true))) {
    APInt AI = FP.bitcastToAPInt();
    return std::string(AI.isNegative() ? "-" : "") + "nan:0x" +
           utohexstr(AI.getZExtValue() &
                         (AI.getBitWidth() == 32 ? INT64_C(0x007fffff)
                                                 : INT64_C(0x000fffffffffffff)),
                     /*LowerCase=*/true);
  }

  // Use C99's hexadecimal floating-point representation.
  static const size_t BufBytes = 128;
  char Buf[BufBytes];
  auto Written = FP.convertToHexString(
      Buf, /*HexDigits=*/0, /*UpperCase=*/false, APFloat::rmNearestTiesToEven);
  (void)Written;
  assert(Written != 0);
  assert(Written < BufBytes);
  return Buf;
}
