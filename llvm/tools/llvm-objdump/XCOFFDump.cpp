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

#include "llvm-objdump.h"
#include "llvm/Demangle/Demangle.h"

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

  std::string SymName = (*SymNameOrErr).str();
  if (Demangle)
    SymName = demangle(SymName);

  if (SymbolDescription)
    SymName = getXCOFFSymbolDescription(createSymbolInfo(Obj, *SymI), SymName);

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

std::string objdump::getXCOFFSymbolDescription(const SymbolInfoTy &SymbolInfo,
                                               StringRef SymbolName) {
  assert(SymbolInfo.isXCOFF() && "Must be a XCOFFSymInfo.");

  std::string Result;
  // Dummy symbols have no symbol index.
  if (SymbolInfo.XCOFFSymInfo.Index)
    Result = ("(idx: " + Twine(SymbolInfo.XCOFFSymInfo.Index.getValue()) +
              ") " + SymbolName)
                 .str();
  else
    Result.append(SymbolName.begin(), SymbolName.end());

  if (SymbolInfo.XCOFFSymInfo.StorageMappingClass &&
      !SymbolInfo.XCOFFSymInfo.IsLabel) {
    const XCOFF::StorageMappingClass Smc =
        SymbolInfo.XCOFFSymInfo.StorageMappingClass.getValue();
    Result.append(("[" + XCOFF::getMappingClassString(Smc) + "]").str());
  }

  return Result;
}
