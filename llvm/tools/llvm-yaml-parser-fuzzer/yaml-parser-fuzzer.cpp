//===-- yaml-parser-fuzzer.cpp - Fuzzer for YAML parser -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLParser.h"

using namespace llvm;

static bool isValidYaml(const uint8_t *Data, size_t Size) {
  SourceMgr SM;
  yaml::Stream Stream(StringRef(reinterpret_cast<const char *>(Data), Size),
                      SM);
  return Stream.validate();
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  std::vector<uint8_t> Input(Data, Data + Size);

  // Ensure we don't crash on byte strings where the only null character is
  // one-past-the-end of the actual input to the parser.
  Input.erase(std::remove(Input.begin(), Input.end(), 0), Input.end());
  Input.push_back(0);
  Input.shrink_to_fit();
  isValidYaml(Input.data(), Input.size() - 1);

  return 0;
}
