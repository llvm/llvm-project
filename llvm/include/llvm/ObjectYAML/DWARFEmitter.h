//===--- DWARFEmitter.h - ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Common declarations for yaml2obj
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_DWARFEMITTER_H
#define LLVM_OBJECTYAML_DWARFEMITTER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include <memory>

namespace llvm {

class raw_ostream;

namespace DWARFYAML {

struct Data;

LLVM_ABI Error emitDebugAbbrev(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugStr(raw_ostream &OS, const Data &DI);

LLVM_ABI Error emitDebugAranges(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugRanges(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugPubnames(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugPubtypes(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugGNUPubnames(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugGNUPubtypes(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugInfo(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugLine(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugAddr(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugStrOffsets(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugRnglists(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugLoclists(raw_ostream &OS, const Data &DI);
LLVM_ABI Error emitDebugNames(raw_ostream &OS, const Data &DI);

LLVM_ABI std::function<Error(raw_ostream &, const Data &)>
getDWARFEmitterByName(StringRef SecName);
LLVM_ABI Expected<StringMap<std::unique_ptr<MemoryBuffer>>>
emitDebugSections(StringRef YAMLString,
                  bool IsLittleEndian = sys::IsLittleEndianHost,
                  bool Is64BitAddrSize = true);
} // end namespace DWARFYAML
} // end namespace llvm

#endif // LLVM_OBJECTYAML_DWARFEMITTER_H
