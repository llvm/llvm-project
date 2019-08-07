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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace llvm {
class raw_ostream;
template <typename T> class SmallVectorImpl;
template <typename T> class Expected;

namespace object {
class ObjectFile;
}

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

int yaml2coff(COFFYAML::Object &Doc, raw_ostream &Out);
int yaml2elf(ELFYAML::Object &Doc, raw_ostream &Out);
int yaml2macho(YamlObjectFile &Doc, raw_ostream &Out);
int yaml2minidump(MinidumpYAML::Object &Doc, raw_ostream &Out);
int yaml2wasm(WasmYAML::Object &Doc, raw_ostream &Out);

Error convertYAML(Input &YIn, raw_ostream &Out, unsigned DocNum = 1);

/// Convenience function for tests.
Expected<std::unique_ptr<object::ObjectFile>>
yaml2ObjectFile(SmallVectorImpl<char> &Storage, StringRef Yaml);

} // namespace yaml
} // namespace llvm

#endif
