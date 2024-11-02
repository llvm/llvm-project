//===- CompilationDatabase.cpp - LSP Compilation Database -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompilationDatabase.h"
#include "../lsp-server-support/Logging.h"
#include "Protocol.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

using namespace mlir;
using namespace mlir::lsp;

//===----------------------------------------------------------------------===//
// YamlFileInfo
//===----------------------------------------------------------------------===//

namespace {
struct YamlFileInfo {
  /// The absolute path to the file.
  std::string filename;
  /// The include directories available for the file.
  std::vector<std::string> includeDirs;
};
} // namespace

//===----------------------------------------------------------------------===//
// CompilationDatabase
//===----------------------------------------------------------------------===//

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(YamlFileInfo)

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<YamlFileInfo> {
  static void mapping(IO &io, YamlFileInfo &info) {
    // Parse the filename and normalize it to the form we will expect from
    // incoming URIs.
    io.mapRequired("filepath", info.filename);

    // Normalize the filename to avoid incompatability with incoming URIs.
    if (Expected<lsp::URIForFile> uri =
            lsp::URIForFile::fromFile(info.filename))
      info.filename = uri->file().str();

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

const CompilationDatabase::FileInfo &
CompilationDatabase::getFileInfo(StringRef filename) const {
  auto it = files.find(filename);
  return it == files.end() ? defaultFileInfo : it->second;
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
  std::vector<YamlFileInfo> parsedFiles;
  yaml >> parsedFiles;

  SetVector<StringRef> knownIncludes;
  for (auto &file : parsedFiles) {
    auto it = files.try_emplace(file.filename, std::move(file.includeDirs));

    // If we encounter a duplicate file, log a warning and ignore it.
    if (!it.second) {
      Logger::info("Duplicate file in compilation database: {0}",
                   file.filename);
      continue;
    }

    // Track the includes for the file.
    for (StringRef include : it.first->second.includeDirs)
      knownIncludes.insert(include);
  }

  // Add all of the known includes to the default file info. We don't know any
  // information about how to treat these files, but these may be project files
  // that we just don't yet have information for. In these cases, providing some
  // heuristic information provides a better user experience, and generally
  // shouldn't lead to any negative side effects.
  for (StringRef include : knownIncludes)
    defaultFileInfo.includeDirs.push_back(include.str());
}
