//===- ObjectYAML.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_OBJECTYAML_H
#define LLVM_OBJECTYAML_OBJECTYAML_H

#include "llvm/ObjectYAML/COFFYAML.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ObjectYAML/MachOYAML.h"
#include "llvm/ObjectYAML/WasmYAML.h"
#include "llvm/Support/YAMLTraits.h"
#include <memory>

namespace llvm {
namespace yaml {

class IO;

struct YamlObjectFile {
  std::unique_ptr<ELFYAML::Object> Elf;
  std::unique_ptr<COFFYAML::Object> Coff;
  std::unique_ptr<MachOYAML::Object> MachO;
  std::unique_ptr<MachOYAML::UniversalBinary> FatMachO;
  std::unique_ptr<WasmYAML::Object> Wasm;
};

template <> struct MappingTraits<YamlObjectFile> {
  static void mapping(IO &IO, YamlObjectFile &ObjectFile);
};

} // end namespace yaml
} // end namespace llvm

#endif // LLVM_OBJECTYAML_OBJECTYAML_H
