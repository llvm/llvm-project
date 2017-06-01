//===--- yaml2obj.h - -------------------------------------------*- C++ -*-===//
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
#ifndef LLVM_TOOLS_YAML2OBJ_YAML2OBJ_H
#define LLVM_TOOLS_YAML2OBJ_YAML2OBJ_H

namespace llvm {
class raw_ostream;

namespace COFFYAML {
struct Object;
}

namespace ELFYAML {
struct Object;
}

namespace DWARFYAML {
struct Data;
struct PubSection;
}

namespace yaml {
class Input;
struct YamlObjectFile;
}
}

int yaml2coff(llvm::COFFYAML::Object &Doc, llvm::raw_ostream &Out);
int yaml2elf(llvm::ELFYAML::Object &Doc, llvm::raw_ostream &Out);
int yaml2macho(llvm::yaml::YamlObjectFile &Doc, llvm::raw_ostream &Out);

void yaml2debug_abbrev(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);
void yaml2debug_str(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);

void yaml2debug_aranges(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);
void yaml2pubsection(llvm::raw_ostream &OS,
                     const llvm::DWARFYAML::PubSection &Sect,
                     bool IsLittleEndian);
void yaml2debug_info(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);
void yaml2debug_line(llvm::raw_ostream &OS, const llvm::DWARFYAML::Data &DI);

#endif
