//===-- TestUtilities.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestUtilities.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBTarget.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <utility>

using namespace lldb_private;

extern const char *TestMainArgv0;

std::once_flag TestUtilities::g_debugger_initialize_flag;

std::string lldb_private::PrettyPrint(const llvm::json::Value &value) {
  return llvm::formatv("{0:2}", value).str();
}

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

bool lldb_private::DebuggerSupportsLLVMTarget(llvm::StringRef target) {
  lldb::SBStructuredData data = lldb::SBDebugger::GetBuildConfiguration()
                                    .GetValueForKey("targets")
                                    .GetValueForKey("value");
  for (size_t i = 0; i < data.GetSize(); i++) {
    char buf[100] = {0};
    size_t size = data.GetItemAtIndex(i).GetStringValue(buf, sizeof(buf));
    if (llvm::StringRef(buf, size) == target)
      return true;
  }

  return false;
}

std::pair<lldb::SBTarget, lldb::SBProcess>
lldb_private::LoadCore(lldb::SBDebugger &debugger, llvm::StringRef binary_path,
                       llvm::StringRef core_path) {
  EXPECT_TRUE(debugger);

  llvm::Expected<lldb_private::TestFile> binary_yaml =
      lldb_private::TestFile::fromYamlFile(binary_path);
  EXPECT_THAT_EXPECTED(binary_yaml, llvm::Succeeded());
  llvm::Expected<llvm::sys::fs::TempFile> binary_file =
      binary_yaml->writeToTemporaryFile();
  EXPECT_THAT_EXPECTED(binary_file, llvm::Succeeded());
  lldb::SBError error;
  lldb::SBTarget target = debugger.CreateTarget(
      /*filename=*/binary_file->TmpName.data(), /*target_triple=*/"",
      /*platform_name=*/"", /*add_dependent_modules=*/false, /*error=*/error);
  EXPECT_TRUE(target);
  EXPECT_TRUE(error.Success()) << error.GetCString();
  debugger.SetSelectedTarget(target);

  llvm::Expected<lldb_private::TestFile> core_yaml =
      lldb_private::TestFile::fromYamlFile(core_path);
  EXPECT_THAT_EXPECTED(core_yaml, llvm::Succeeded());
  llvm::Expected<llvm::sys::fs::TempFile> core_file =
      core_yaml->writeToTemporaryFile();
  EXPECT_THAT_EXPECTED(core_file, llvm::Succeeded());
  lldb::SBProcess process = target.LoadCore(core_file->TmpName.data());
  EXPECT_TRUE(process);

  EXPECT_THAT_ERROR(binary_file->discard(), llvm::Succeeded());
  EXPECT_THAT_ERROR(core_file->discard(), llvm::Succeeded());

  return std::make_pair(target, process);
}
