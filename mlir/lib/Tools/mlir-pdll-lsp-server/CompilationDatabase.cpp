//===- CompilationDatabase.cpp - PDLL Compilation Database ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompilationDatabase.h"
#include "../lsp-server-support/Logging.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

using namespace mlir;
using namespace mlir::lsp;

//===----------------------------------------------------------------------===//
// CompilationDatabase
//===----------------------------------------------------------------------===//

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(CompilationDatabase::FileInfo)

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<CompilationDatabase::FileInfo> {
  static void mapping(IO &io, CompilationDatabase::FileInfo &info) {
    io.mapRequired("filepath", info.filename);

    // Parse the includes from the yaml stream. These are in the form of a
    // semi-colon delimited list.
    std::string combinedIncludes;
    io.mapRequired("includes", combinedIncludes);
    for (StringRef include : llvm::split(combinedIncludes, ";")) {
      if (!include.empty())
        info.includeDirs.push_back(include.str());
    }
  }
};
} // end namespace yaml
} // end namespace llvm

CompilationDatabase::CompilationDatabase(ArrayRef<std::string> databases) {
  for (StringRef filename : databases)
    loadDatabase(filename);
}

const CompilationDatabase::FileInfo *
CompilationDatabase::getFileInfo(StringRef filename) const {
  auto it = files.find(filename);
  return it == files.end() ? nullptr : &it->second;
}

void CompilationDatabase::loadDatabase(StringRef filename) {
  if (filename.empty())
    return;

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      openInputFile(filename, &errorMessage);
  if (!inputFile) {
    Logger::error("Failed to open compilation database: {0}", errorMessage);
    return;
  }
  llvm::yaml::Input yaml(inputFile->getBuffer());

  // Parse the yaml description and add any new files to the database.
  std::vector<FileInfo> parsedFiles;
  yaml >> parsedFiles;
  for (auto &file : parsedFiles) {
    auto it = files.try_emplace(file.filename, std::move(file));

    // If we encounter a duplicate file, log a warning and ignore it.
    if (!it.second) {
      Logger::info("Duplicate .pdll file in compilation database: {0}",
                   file.filename);
    }
  }
}
