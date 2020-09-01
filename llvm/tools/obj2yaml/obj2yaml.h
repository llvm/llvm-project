//===------ utils/obj2yaml.hpp - obj2yaml conversion tool -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file declares some helper routines, and also the format-specific
// writers. To add a new format, add the declaration here, and, in a separate
// source file, implement it.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJ2YAML_OBJ2YAML_H
#define LLVM_TOOLS_OBJ2YAML_OBJ2YAML_H

#include "llvm/Object/COFF.h"
#include "llvm/Object/Minidump.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

std::error_code coff2yaml(llvm::raw_ostream &Out,
                          const llvm::object::COFFObjectFile &Obj);
llvm::Error elf2yaml(llvm::raw_ostream &Out,
                         const llvm::object::ObjectFile &Obj);
llvm::Error macho2yaml(llvm::raw_ostream &Out,
                           const llvm::object::Binary &Obj);
llvm::Error minidump2yaml(llvm::raw_ostream &Out,
                          const llvm::object::MinidumpFile &Obj);
std::error_code xcoff2yaml(llvm::raw_ostream &Out,
                           const llvm::object::XCOFFObjectFile &Obj);
std::error_code wasm2yaml(llvm::raw_ostream &Out,
                          const llvm::object::WasmObjectFile &Obj);

// Forward decls for dwarf2yaml
namespace llvm {
class DWARFContext;
namespace DWARFYAML {
struct Data;
}
}

void dumpDebugAbbrev(llvm::DWARFContext &DCtx, llvm::DWARFYAML::Data &Y);
llvm::Error dumpDebugARanges(llvm::DWARFContext &DCtx,
                             llvm::DWARFYAML::Data &Y);
void dumpDebugPubSections(llvm::DWARFContext &DCtx, llvm::DWARFYAML::Data &Y);
void dumpDebugInfo(llvm::DWARFContext &DCtx, llvm::DWARFYAML::Data &Y);
void dumpDebugLines(llvm::DWARFContext &DCtx, llvm::DWARFYAML::Data &Y);
llvm::Error dumpDebugRanges(llvm::DWARFContext &DCtx, llvm::DWARFYAML::Data &Y);
void dumpDebugStrings(llvm::DWARFContext &DCtx, llvm::DWARFYAML::Data &Y);

#endif
