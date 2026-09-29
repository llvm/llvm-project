//===- YAML2ObjTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace object;
using namespace yaml;

TEST(yaml2ObjectFile, ELF) {
  bool ErrorReported = false;
  auto ErrHandler = [&](const Twine &Msg) { ErrorReported = true; };

  SmallString<0> Storage;
  std::unique_ptr<ObjectFile> Obj = yaml2ObjectFile(Storage, R"(
--- !ELF
FileHeader:
  Class:    ELFCLASS64
  Data:     ELFDATA2LSB
  Type:     ET_REL
  Machine:  EM_X86_64)", ErrHandler);

  ASSERT_FALSE(ErrorReported);
  ASSERT_TRUE(Obj);
  ASSERT_TRUE(Obj->isELF());
  ASSERT_TRUE(Obj->isRelocatableObject());
}

TEST(yaml2ObjectFile, Errors) {
  std::vector<std::string> Errors;
  auto ErrHandler = [&](const Twine &Msg) {
    Errors.push_back("ObjectYAML: " + Msg.str());
  };

  SmallString<0> Storage;
  StringRef Yaml = R"(
--- !ELF
FileHeader:
  Class:    ELFCLASS64
  Data:     ELFDATA2LSB
  Type:     ET_REL
  Machine:  EM_X86_64
Symbols:
  - Name: foo
  - Name: foo
  - Name: foo
)";

  // 1. Test yaml2ObjectFile().

  std::unique_ptr<ObjectFile> Obj = yaml2ObjectFile(Storage, Yaml, ErrHandler);

  ASSERT_FALSE(Obj);
  ASSERT_TRUE(Errors.size() == 2);
  ASSERT_TRUE(Errors[0] == "ObjectYAML: repeated symbol name: 'foo'");
  ASSERT_TRUE(Errors[1] == Errors[0]);

  // 2. Test convertYAML(). 

  Errors.clear();
  Storage.clear();
  raw_svector_ostream OS(Storage);

  yaml::Input YIn(Yaml);
  bool Res = convertYAML(YIn, OS, ErrHandler);
  ASSERT_FALSE(Res);
  ASSERT_TRUE(Errors.size() == 2);
  ASSERT_TRUE(Errors[0] == "ObjectYAML: repeated symbol name: 'foo'");
  ASSERT_TRUE(Errors[1] == Errors[0]);
}
