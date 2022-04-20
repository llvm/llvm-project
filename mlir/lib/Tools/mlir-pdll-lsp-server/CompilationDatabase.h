//===- CompilationDatabase.h - PDLL Compilation Database --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRPDLLSPSERVER_COMPILATIONDATABASE_H_
#define LIB_MLIR_TOOLS_MLIRPDLLSPSERVER_COMPILATIONDATABASE_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <string>

namespace mlir {
namespace lsp {
/// This class contains a collection of compilation information for files
/// provided to the language server, such as the available include directories.
/// This database acts as an aggregate in-memory form of compilation databases
/// used by the current language client. The textual form of a compilation
/// database is a YAML file containing documents of the following form:
///
/// --- !FileInfo:
///   filepath: <string> - Absolute file path of the file.
///   includes: <string> - Semi-colon delimited list of include directories.
///
class CompilationDatabase {
public:
  /// Compilation information for a specific file within the database.
  struct FileInfo {
    /// The absolute path to the file.
    std::string filename;
    /// The include directories available for the file.
    std::vector<std::string> includeDirs;
  };

  /// Construct a compilation database from the provided files containing YAML
  /// descriptions of the database.
  CompilationDatabase(ArrayRef<std::string> databases);

  /// Get the compilation information for the provided file, or nullptr if the
  /// database doesn't include information for `filename`.
  const FileInfo *getFileInfo(StringRef filename) const;

private:
  /// Load the given database file into this database.
  void loadDatabase(StringRef filename);

  /// A map of filename to file information for each known file within the
  /// databases.
  llvm::StringMap<FileInfo> files;
};
} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_MLIRPDLLSPSERVER_COMPILATIONDATABASE_H_
