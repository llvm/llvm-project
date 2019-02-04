//===-- WasmDump.cpp - wasm-specific dumper ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the wasm-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/Object/Wasm.h"

using namespace llvm;
using namespace object;

void llvm::printWasmFileHeader(const object::ObjectFile *Obj) {
  const WasmObjectFile *File = dyn_cast<const WasmObjectFile>(Obj);

  outs() << "Program Header:\n";
  outs() << "Version: 0x";
  outs().write_hex(File->getHeader().Version);
  outs() << "\n";
}

std::error_code
llvm::getWasmRelocationValueString(const WasmObjectFile *Obj,
                                   const RelocationRef &RelRef,
                                   SmallVectorImpl<char> &Result) {
  const wasm::WasmRelocation &Rel = Obj->getWasmRelocation(RelRef);
  symbol_iterator SI = RelRef.getSymbol();
  std::string FmtBuf;
  raw_string_ostream Fmt(FmtBuf);
  if (SI == Obj->symbol_end()) {
    // Not all wasm relocations have symbols associated with them.
    // In particular R_WASM_TYPE_INDEX_LEB.
    Fmt << Rel.Index;
  } else {
    Expected<StringRef> SymNameOrErr = SI->getName();
    if (!SymNameOrErr)
      return errorToErrorCode(SymNameOrErr.takeError());
    StringRef SymName = *SymNameOrErr;
    Result.append(SymName.begin(), SymName.end());
  }
  Fmt << (Rel.Addend < 0 ? "" : "+") << Rel.Addend;
  Fmt.flush();
  Result.append(FmtBuf.begin(), FmtBuf.end());
  return std::error_code();
}
