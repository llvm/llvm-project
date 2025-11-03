//===-- TestUtilities.cpp -------------------------------------------------===//
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

std::once_flag TestUtilities::g_debugger_initialize_flag;
std::string lldb_private::GetInputFilePath(const llvm::Twine &name) {
  llvm::SmallString<128> result = llvm::sys::path::parent_path(TestMainArgv0);
  llvm::sys::fs::make_absolute(result);
  llvm::sys::path::append(result, "Inputs", name);
  return std::string(result.str());
}

llvm::Expected<TestFile> TestFile::fromYaml(llvm::StringRef Yaml) {
  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  llvm::yaml::Input YIn(Yaml);
  std::string ErrorMsg("convertYAML() failed: ");
  if (!llvm::yaml::convertYAML(YIn, OS, [&ErrorMsg](const llvm::Twine &Msg) {
        ErrorMsg += Msg.str();
      }))
    return llvm::createStringError(llvm::inconvertibleErrorCode(), ErrorMsg);
  return TestFile(std::move(Buffer));
}

llvm::Expected<TestFile> TestFile::fromYamlFile(const llvm::Twine &Name) {
  auto BufferOrError =
      llvm::MemoryBuffer::getFile(GetInputFilePath(Name), /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!BufferOrError)
    return llvm::errorCodeToError(BufferOrError.getError());
  return fromYaml(BufferOrError.get()->getBuffer());
}

llvm::Expected<llvm::sys::fs::TempFile> TestFile::writeToTemporaryFile() {
  llvm::Expected<llvm::sys::fs::TempFile> Temp =
      llvm::sys::fs::TempFile::create("temp%%%%%%%%%%%%%%%%");
  if (!Temp)
    return Temp.takeError();
  llvm::raw_fd_ostream(Temp->FD, /*shouldClose=*/false) << Buffer;
  return std::move(*Temp);
}
