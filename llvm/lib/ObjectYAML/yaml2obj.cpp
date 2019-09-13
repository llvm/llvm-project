//===-- yaml2obj.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ObjectYAML/ObjectYAML.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace yaml {

Error convertYAML(yaml::Input &YIn, raw_ostream &Out, unsigned DocNum) {
  auto BoolToErr = [](bool Ret) -> Error {
    if (!Ret)
      return createStringError(errc::invalid_argument, "yaml2obj failed");
    return Error::success();
  };

  unsigned CurDocNum = 0;
  do {
    if (++CurDocNum == DocNum) {
      yaml::YamlObjectFile Doc;
      YIn >> Doc;
      if (std::error_code EC = YIn.error())
        return createStringError(EC, "Failed to parse YAML input!");
      if (Doc.Elf)
        return BoolToErr(yaml2elf(*Doc.Elf, Out));
      if (Doc.Coff)
        return BoolToErr(yaml2coff(*Doc.Coff, Out));
      if (Doc.MachO || Doc.FatMachO)
        return BoolToErr(yaml2macho(Doc, Out));
      if (Doc.Minidump)
        return BoolToErr(yaml2minidump(*Doc.Minidump, Out));
      if (Doc.Wasm)
        return BoolToErr(yaml2wasm(*Doc.Wasm, Out));
      return createStringError(errc::invalid_argument,
                               "Unknown document type!");
    }
  } while (YIn.nextDocument());

  return createStringError(errc::invalid_argument,
                           "Cannot find the %u%s document", DocNum,
                           getOrdinalSuffix(DocNum).data());
}

Expected<std::unique_ptr<object::ObjectFile>>
yaml2ObjectFile(SmallVectorImpl<char> &Storage, StringRef Yaml) {
  Storage.clear();
  raw_svector_ostream OS(Storage);

  yaml::Input YIn(Yaml);
  if (Error E = convertYAML(YIn, OS))
    return std::move(E);

  return object::ObjectFile::createObjectFile(
      MemoryBufferRef(OS.str(), "YamlObject"));
}

} // namespace yaml
} // namespace llvm
