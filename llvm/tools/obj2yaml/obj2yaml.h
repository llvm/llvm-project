//===------ utils/obj2yaml.hpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// This file declares some helper routines, and also the format-specific
// writers. To add a new format, add the declaration here, and, in a separate
// source file, implement it.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJ2YAML_OBJ2YAML_H
#define LLVM_TOOLS_OBJ2YAML_OBJ2YAML_H

#include "llvm/Object/COFF.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

std::error_code coff2yaml(llvm::raw_ostream &Out,
                          const llvm::object::COFFObjectFile &Obj);
std::error_code elf2yaml(llvm::raw_ostream &Out,
                         const llvm::object::ObjectFile &Obj);
std::error_code macho2yaml(llvm::raw_ostream &Out,
                           const llvm::object::Binary &Obj);
std::error_code wasm2yaml(llvm::raw_ostream &Out,
                          const llvm::object::WasmObjectFile &Obj);

// Forward decls for dwarf2yaml
namespace llvm {
class DWARFContextInMemory;
namespace DWARFYAML {
struct Data;
}
}

std::error_code dwarf2yaml(llvm::DWARFContextInMemory &DCtx,
                           llvm::DWARFYAML::Data &Y);

#endif
