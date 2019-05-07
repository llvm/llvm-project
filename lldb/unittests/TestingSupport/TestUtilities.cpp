//===- TestUtilities.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestUtilities.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

extern const char *TestMainArgv0;

std::string lldb_private::GetInputFilePath(const llvm::Twine &name) {
  llvm::SmallString<128> result = llvm::sys::path::parent_path(TestMainArgv0);
  llvm::sys::fs::make_absolute(result);
  llvm::sys::path::append(result, "Inputs", name);
  return result.str();
}

llvm::Error
lldb_private::ReadYAMLObjectFile(const llvm::Twine &yaml_name,
                                 llvm::SmallString<128> &object_file) {
  std::string yaml = GetInputFilePath(yaml_name);
  llvm::StringRef args[] = {YAML2OBJ, yaml};
  llvm::StringRef obj_ref = object_file;
  const llvm::Optional<llvm::StringRef> redirects[] = {llvm::None, obj_ref,
                                                       llvm::None};
  if (llvm::sys::ExecuteAndWait(YAML2OBJ, args, llvm::None, redirects) != 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Error running yaml2obj %s.", yaml.c_str());
  uint64_t size;
  if (auto ec = llvm::sys::fs::file_size(object_file, size))
    return llvm::errorCodeToError(ec);
  if (size == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Empty object file created from yaml2obj %s.", yaml.c_str());
  return llvm::Error::success();
}