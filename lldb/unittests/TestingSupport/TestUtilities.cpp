//===- TestUtilities.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestUtilities.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/YAMLTraits.h"
#include "gtest/gtest.h"

using namespace lldb_private;

extern const char *TestMainArgv0;

std::string lldb_private::GetInputFilePath(const llvm::Twine &name) {
  llvm::SmallString<128> result = llvm::sys::path::parent_path(TestMainArgv0);
  llvm::sys::fs::make_absolute(result);
  llvm::sys::path::append(result, "Inputs", name);
  return result.str();
}

llvm::Expected<TestFile> TestFile::fromYaml(llvm::StringRef Yaml) {
  const auto *Info = testing::UnitTest::GetInstance()->current_test_info();
  assert(Info);
  llvm::SmallString<128> Name;
  int FD;
  if (std::error_code EC = llvm::sys::fs::createTemporaryFile(
          llvm::Twine(Info->test_case_name()) + "-" + Info->name(), "test", FD,
          Name))
    return llvm::errorCodeToError(EC);
  llvm::FileRemover Remover(Name);
  {
    llvm::raw_fd_ostream OS(FD, /*shouldClose*/ true);
    llvm::yaml::Input YIn(Yaml);
    if (!llvm::yaml::convertYAML(YIn, OS, [](const llvm::Twine &Msg) {}))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "convertYAML() failed");
  }
  return TestFile(Name, std::move(Remover));
}

llvm::Expected<TestFile> TestFile::fromYamlFile(const llvm::Twine &Name) {
  auto BufferOrError =
      llvm::MemoryBuffer::getFile(GetInputFilePath(Name), /*FileSize*/ -1,
                                  /*RequiresNullTerminator*/ false);
  if (!BufferOrError)
    return llvm::errorCodeToError(BufferOrError.getError());
  return fromYaml(BufferOrError.get()->getBuffer());
}

TestFile::~TestFile() {
  if (!Name)
    return;
  if (std::error_code EC =
          llvm::sys::fs::remove(*Name, /*IgnoreNonExisting*/ false))
    GTEST_LOG_(WARNING) << "Failed to delete `" << Name->c_str()
                        << "`: " << EC.message();
}
