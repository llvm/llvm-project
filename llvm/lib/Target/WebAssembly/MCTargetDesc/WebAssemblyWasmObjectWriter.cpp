//===-- WebAssemblyWasmObjectWriter.cpp - WebAssembly Wasm Writer ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file handles Wasm-specific object emission, converting LLVM's
/// internal fixups into the appropriate relocations.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyFixupKinds.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/MC/MCWasmObjectWriter.h"
#include "llvm/MC/MCValue.h"
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

// Test whether the given expression computes a function address.
static bool IsFunctionExpr(const MCExpr *Expr) {
  if (auto SyExp = dyn_cast<MCSymbolRefExpr>(Expr))
    return cast<MCSymbolWasm>(SyExp->getSymbol()).isFunction();

  if (auto BinOp = dyn_cast<MCBinaryExpr>(Expr))
    return IsFunctionExpr(BinOp->getLHS()) != IsFunctionExpr(BinOp->getRHS());

  if (auto UnOp = dyn_cast<MCUnaryExpr>(Expr))
    return IsFunctionExpr(UnOp->getSubExpr());

  return false;
}

static bool IsFunctionType(const MCValue &Target) {
  const MCSymbolRefExpr *RefA = Target.getSymA();
  return RefA && RefA->getKind() == MCSymbolRefExpr::VK_WebAssembly_TYPEINDEX;
}

unsigned
WebAssemblyWasmObjectWriter::getRelocType(const MCValue &Target,
                                          const MCFixup &Fixup) const {
  // WebAssembly functions are not allocated in the data address space. To
  // resolve a pointer to a function, we must use a special relocation type.
  bool IsFunction = IsFunctionExpr(Fixup.getValue());

  switch (unsigned(Fixup.getKind())) {
  case WebAssembly::fixup_code_global_index:
    return wasm::R_WEBASSEMBLY_GLOBAL_INDEX_LEB;
  case WebAssembly::fixup_code_sleb128_i32:
    if (IsFunction)
      return wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB;
    return wasm::R_WEBASSEMBLY_MEMORY_ADDR_SLEB;
  case WebAssembly::fixup_code_sleb128_i64:
    llvm_unreachable("fixup_sleb128_i64 not implemented yet");
  case WebAssembly::fixup_code_uleb128_i32:
    if (IsFunctionType(Target))
      return wasm::R_WEBASSEMBLY_TYPE_INDEX_LEB;
    if (IsFunction)
      return wasm::R_WEBASSEMBLY_FUNCTION_INDEX_LEB;
    return wasm::R_WEBASSEMBLY_MEMORY_ADDR_LEB;
  case FK_Data_4:
    if (IsFunction)
      return wasm::R_WEBASSEMBLY_TABLE_INDEX_I32;
    return wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32;
  case FK_Data_8:
    llvm_unreachable("FK_Data_8 not implemented yet");
  default:
    llvm_unreachable("unimplemented fixup kind");
  }
}

MCObjectWriter *llvm::createWebAssemblyWasmObjectWriter(raw_pwrite_stream &OS,
                                                        bool Is64Bit) {
  MCWasmObjectTargetWriter *MOTW = new WebAssemblyWasmObjectWriter(Is64Bit);
  return createWasmObjectWriter(MOTW, OS);
}
