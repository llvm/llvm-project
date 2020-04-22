//===-- XCOFFDump.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_OBJDUMP_XCOFFDUMP_H
#define LLVM_TOOLS_LLVM_OBJDUMP_XCOFFDUMP_H

#include "llvm/Object/XCOFFObjectFile.h"

namespace llvm {

struct SymbolInfoTy;

namespace objdump {
Optional<XCOFF::StorageMappingClass>
getXCOFFSymbolCsectSMC(const object::XCOFFObjectFile *Obj,
                       const object::SymbolRef &Sym);

bool isLabel(const object::XCOFFObjectFile *Obj, const object::SymbolRef &Sym);

std::string getXCOFFSymbolDescription(const SymbolInfoTy &SymbolInfo,
                                      StringRef SymbolName);

Error getXCOFFRelocationValueString(const object::XCOFFObjectFile *Obj,
                                    const object::RelocationRef &RelRef,
                                    llvm::SmallVectorImpl<char> &Result);
} // namespace objdump
} // namespace llvm
#endif
