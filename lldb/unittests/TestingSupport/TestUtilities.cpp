//===- TestUtilities.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestUtilities.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

extern const char *TestMainArgv0;

std::string lldb_private::GetInputFilePath(const llvm::Twine &name) {
  llvm::SmallString<128> result = llvm::sys::path::parent_path(TestMainArgv0);
  llvm::sys::fs::make_absolute(result);
  llvm::sys::path::append(result, "Inputs", name);
  return result.str();
}
