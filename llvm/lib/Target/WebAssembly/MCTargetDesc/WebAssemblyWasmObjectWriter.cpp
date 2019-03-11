//===-- WebAssemblyWasmObjectWriter.cpp - WebAssembly Wasm Writer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file handles Wasm-specific object emission, converting LLVM's
/// internal fixups into the appropriate relocations.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyFixupKinds.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWasmObjectWriter.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class WebAssemblyWasmObjectWriter final : public MCWasmObjectTargetWriter {
public:
  explicit WebAssemblyWasmObjectWriter(bool Is64Bit);

private:
  unsigned getRelocType(const MCValue &Target,
                        const MCFixup &Fixup) const override;
};
} // end anonymous namespace

WebAssemblyWasmObjectWriter::WebAssemblyWasmObjectWriter(bool Is64Bit)
    : MCWasmObjectTargetWriter(Is64Bit) {}

static bool isFunctionSignatureRef(const MCSymbolRefExpr *Ref) {
  return Ref->getKind() == MCSymbolRefExpr::VK_WebAssembly_TYPEINDEX;
}

static const MCSection *getFixupSection(const MCExpr *Expr) {
  if (auto SyExp = dyn_cast<MCSymbolRefExpr>(Expr)) {
    if (SyExp->getSymbol().isInSection())
      return &SyExp->getSymbol().getSection();
    return nullptr;
  }

  if (auto BinOp = dyn_cast<MCBinaryExpr>(Expr)) {
    auto SectionLHS = getFixupSection(BinOp->getLHS());
    auto SectionRHS = getFixupSection(BinOp->getRHS());
    return SectionLHS == SectionRHS ? nullptr : SectionLHS;
  }

  if (auto UnOp = dyn_cast<MCUnaryExpr>(Expr))
    return getFixupSection(UnOp->getSubExpr());

  return nullptr;
}

unsigned WebAssemblyWasmObjectWriter::getRelocType(const MCValue &Target,
                                                   const MCFixup &Fixup) const {
  const MCSymbolRefExpr *RefA = Target.getSymA();
  assert(RefA);
  auto& SymA = cast<MCSymbolWasm>(RefA->getSymbol());

  switch (unsigned(Fixup.getKind())) {
  case WebAssembly::fixup_code_sleb128_i32:
    if (SymA.isFunction())
      return wasm::R_WASM_TABLE_INDEX_SLEB;
    return wasm::R_WASM_MEMORY_ADDR_SLEB;
  case WebAssembly::fixup_code_sleb128_i64:
    llvm_unreachable("fixup_sleb128_i64 not implemented yet");
  case WebAssembly::fixup_code_uleb128_i32:
    if (SymA.isFunction()) {
      if (isFunctionSignatureRef(RefA))
        return wasm::R_WASM_TYPE_INDEX_LEB;
      else
        return wasm::R_WASM_FUNCTION_INDEX_LEB;
    }
    if (SymA.isGlobal())
      return wasm::R_WASM_GLOBAL_INDEX_LEB;
    if (SymA.isEvent())
      return wasm::R_WASM_EVENT_INDEX_LEB;
    return wasm::R_WASM_MEMORY_ADDR_LEB;
  case FK_Data_4:
    if (SymA.isFunction())
      return wasm::R_WASM_TABLE_INDEX_I32;
    if (auto Section = static_cast<const MCSectionWasm *>(
            getFixupSection(Fixup.getValue()))) {
      if (Section->getKind().isText())
        return wasm::R_WASM_FUNCTION_OFFSET_I32;
      else if (!Section->isWasmData())
        return wasm::R_WASM_SECTION_OFFSET_I32;
    }
    return wasm::R_WASM_MEMORY_ADDR_I32;
  case FK_Data_8:
    llvm_unreachable("FK_Data_8 not implemented yet");
  default:
    llvm_unreachable("unimplemented fixup kind");
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createWebAssemblyWasmObjectWriter(bool Is64Bit) {
  return llvm::make_unique<WebAssemblyWasmObjectWriter>(Is64Bit);
}
