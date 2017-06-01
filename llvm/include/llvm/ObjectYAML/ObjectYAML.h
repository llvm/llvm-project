//===- ObjectYAML.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_OBJECTYAML_H
#define LLVM_OBJECTYAML_OBJECTYAML_H

#include "llvm/Support/YAMLTraits.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ObjectYAML/COFFYAML.h"
#include "llvm/ObjectYAML/MachOYAML.h"

namespace llvm {
namespace yaml {

struct YamlObjectFile {
  std::unique_ptr<ELFYAML::Object> Elf;
  std::unique_ptr<COFFYAML::Object> Coff;
  std::unique_ptr<MachOYAML::Object> MachO;
  std::unique_ptr<MachOYAML::UniversalBinary> FatMachO;
};

template <> struct MappingTraits<YamlObjectFile> {
  static void mapping(IO &IO, YamlObjectFile &ObjectFile);
};

} // namespace yaml
} // namespace llvm

#endif
