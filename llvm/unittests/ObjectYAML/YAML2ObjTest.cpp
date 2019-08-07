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
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace object;
using namespace yaml;

TEST(yaml2ObjectFile, ELF) {
  SmallString<0> Storage;
  Expected<std::unique_ptr<ObjectFile>> ErrOrObj = yaml2ObjectFile(Storage, R"(
--- !ELF
FileHeader:
  Class:    ELFCLASS64
  Data:     ELFDATA2LSB
  Type:     ET_REL
  Machine:  EM_X86_64)");

  ASSERT_THAT_EXPECTED(ErrOrObj, Succeeded());

  std::unique_ptr<ObjectFile> ObjFile = std::move(ErrOrObj.get());

  ASSERT_TRUE(ObjFile->isELF());
  ASSERT_TRUE(ObjFile->isRelocatableObject());
}
