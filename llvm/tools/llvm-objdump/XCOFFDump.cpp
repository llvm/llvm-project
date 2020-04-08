//===-- XCOFFDump.cpp - XCOFF-specific dumper -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XCOFF-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "XCOFFDump.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"

using namespace llvm;
using namespace llvm::object;

Error objdump::getXCOFFRelocationValueString(const XCOFFObjectFile *Obj,
                                             const RelocationRef &Rel,
                                             SmallVectorImpl<char> &Result) {
  symbol_iterator SymI = Rel.getSymbol();
  if (SymI == Obj->symbol_end())
    return make_error<GenericBinaryError>(
        "invalid symbol reference in relocation entry",
        object_error::parse_failed);

  Expected<StringRef> SymNameOrErr = SymI->getName();
  if (!SymNameOrErr)
    return SymNameOrErr.takeError();
  StringRef SymName = *SymNameOrErr;
  Result.append(SymName.begin(), SymName.end());
  return Error::success();
}

Optional<XCOFF::StorageMappingClass>
objdump::getXCOFFSymbolCsectSMC(const XCOFFObjectFile *Obj,
                                const SymbolRef &Sym) {
  XCOFFSymbolRef SymRef(Sym.getRawDataRefImpl(), Obj);

  if (SymRef.hasCsectAuxEnt())
    return SymRef.getXCOFFCsectAuxEnt32()->StorageMappingClass;

  return None;
}

bool objdump::isLabel(const XCOFFObjectFile *Obj, const SymbolRef &Sym) {

  XCOFFSymbolRef SymRef(Sym.getRawDataRefImpl(), Obj);

  if (SymRef.hasCsectAuxEnt())
    return SymRef.getXCOFFCsectAuxEnt32()->isLabel();

  return false;
}

void objdump::printXCOFFSymbolDescription(const SymbolInfoTy &SymbolInfo,
                                          StringRef SymbolName) {
  assert(SymbolInfo.isXCOFF() && "Must be a XCOFFSymInfo.");

  // Dummy symbols have no symbol index.
  if (SymbolInfo.XCOFFSymInfo.Index)
    outs() << "(idx: " << SymbolInfo.XCOFFSymInfo.Index.getValue() << ") ";

  outs() << SymbolName;

  if (SymbolInfo.XCOFFSymInfo.StorageMappingClass &&
      !SymbolInfo.XCOFFSymInfo.IsLabel) {
    const XCOFF::StorageMappingClass Smc =
        SymbolInfo.XCOFFSymInfo.StorageMappingClass.getValue();
    outs() << "[" << XCOFF::getMappingClassString(Smc) << "]";
  }
}
