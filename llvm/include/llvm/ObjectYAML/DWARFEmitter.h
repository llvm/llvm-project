//===--- DWARFEmitter.h - -------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief Common declarations for yaml2obj
//===----------------------------------------------------------------------===//
#ifndef LLVM_OBJECTYAML_DWARFEMITTER_H
#define LLVM_OBJECTYAML_DWARFEMITTER_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <vector>

namespace llvm {
class raw_ostream;

namespace DWARFYAML {
struct Data;
struct PubSection;

void EmitDebugAbbrev(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);
void EmitDebugStr(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);

void EmitDebugAranges(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);
void EmitPubSection(llvm::raw_ostream &OS,
                    const llvm::DWARFYAML::PubSection &Sect,
                    bool IsLittleEndian);
void EmitDebugInfo(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);
void EmitDebugLine(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);

Expected<StringMap<std::unique_ptr<MemoryBuffer>>>
EmitDebugSections(StringRef YAMLString,
                  bool IsLittleEndian = sys::IsLittleEndianHost);

} // namespace DWARFYAML
} // namespace llvm

#endif
