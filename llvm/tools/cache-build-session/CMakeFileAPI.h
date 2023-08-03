//===-- CMakeFileAPI.h - CMake File API -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// APIs to read information from CMake's file-based API. See
// https://cmake.org/cmake/help/latest/manual/cmake-file-api.7.html for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_CACHEBUILDSESSION_CMAKEFILEAPI_H
#define LLVM_TOOLS_CACHEBUILDSESSION_CMAKEFILEAPI_H

#include "llvm/Support/JSON.h"

namespace llvm {
namespace cmake_file_api {

class CodeModel {
  json::Object Obj;

public:
  explicit CodeModel(json::Object Obj) : Obj(std::move(Obj)) {}
  CodeModel() = default;

  /// The CMake build directory.
  Expected<StringRef> getBuildPath() const;

  /// The source directory of the top-level "CMakeLists.txt" file where
  /// configuration was initiated from.
  Expected<StringRef> getSourcePath() const;

  /// Returns an array of other top-level directories that are not contained in
  /// the directory of \p getSourcePath().
  ///
  /// For example, when configuring llvm with
  /// \p -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" the returned top-level
  /// paths will be "/path/to/clang", and "/path/to/clang-tools-extra", with
  /// \p getSourcePath() returning "/path/to/llvm".
  Error getExtraTopLevelSourcePaths(SmallVectorImpl<StringRef> &Paths) const;
};

class Index {
  json::Object Obj;
  std::string CMakeFileAPIPath;

  Index(json::Object Obj, std::string CMakeFileAPIPath)
      : Obj(std::move(Obj)), CMakeFileAPIPath(std::move(CMakeFileAPIPath)) {}

public:
  Index() = default;

  static Expected<Index> fromPath(StringRef CMakeBuildPath);

  Expected<CodeModel> getCodeModel() const;
};

} // namespace cmake_file_api
} // namespace llvm

#endif
