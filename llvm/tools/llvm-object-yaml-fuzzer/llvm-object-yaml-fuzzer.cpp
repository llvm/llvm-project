//===-- llvm-object-yaml-fuzzer.cpp - Fuzzer for llvm/lib/ObjectYaml ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ObjectYAML/yaml2obj.h"

using namespace llvm;
using namespace object;
using namespace yaml;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  bool ErrorReported = false;
  auto ErrHandler = [&](const Twine &Msg) { ErrorReported = true; };
  std::string Payload(reinterpret_cast<const char *>(Data), Size);
  SmallString<0> Storage;
  std::unique_ptr<ObjectFile> Obj =
      yaml2ObjectFile(Storage, Payload, ErrHandler);
  return 0;
}
