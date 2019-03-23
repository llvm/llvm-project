//===--- yaml2obj.h - -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Common declarations for yaml2obj
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

namespace MinidumpYAML {
struct Object;
}

namespace WasmYAML {
struct Object;
}

namespace yaml {
class Input;
struct YamlObjectFile;
}
}

int yaml2coff(llvm::COFFYAML::Object &Doc, llvm::raw_ostream &Out);
int yaml2elf(llvm::ELFYAML::Object &Doc, llvm::raw_ostream &Out);
int yaml2macho(llvm::yaml::YamlObjectFile &Doc, llvm::raw_ostream &Out);
int yaml2minidump(llvm::MinidumpYAML::Object &Doc, llvm::raw_ostream &Out);
int yaml2wasm(llvm::WasmYAML::Object &Doc, llvm::raw_ostream &Out);

#endif
