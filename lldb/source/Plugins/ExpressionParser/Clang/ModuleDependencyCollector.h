//===-- ModuleDependencyCollector.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ModuleDependencyCollector_h_
#define liblldb_ModuleDependencyCollector_h_

#include "lldb/Utility/FileCollector.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {
class ModuleDependencyCollectorAdaptor
    : public clang::ModuleDependencyCollector {
public:
  ModuleDependencyCollectorAdaptor(FileCollector &file_collector)
      : clang::ModuleDependencyCollector(""), m_file_collector(file_collector) {
  }

  void addFile(llvm::StringRef Filename,
               llvm::StringRef FileDst = {}) override {
    m_file_collector.AddFile(Filename);
  }

  bool insertSeen(llvm::StringRef Filename) override { return false; }
  void addFileMapping(llvm::StringRef VPath, llvm::StringRef RPath) override {}
  void writeFileMap() override {}

private:
  FileCollector &m_file_collector;
};
} // namespace lldb_private

#endif
