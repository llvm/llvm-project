//===------------------- FileClassifier.cpp - LLVM Advisor ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the FileClassifier code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "FileClassifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace llvm {
namespace advisor {

FileClassification
FileClassifier::classifyFile(llvm::StringRef FilePath) const {
  StringRef Filename = sys::path::filename(FilePath);
  StringRef Extension = sys::path::extension(FilePath);

  FileClassification Classification;
  Classification.isGenerated = true;
  Classification.isTemporary = false;

  // LLVM IR files
  if (Extension == ".ll") {
    Classification.category = "ir";
    Classification.description = "LLVM IR text";
    return Classification;
  }

  // Assembly files
  if (Extension == ".s" || Extension == ".S") {
    Classification.category = "assembly";
    Classification.description = "Assembly";
    return Classification;
  }

  // Optimization remarks
  if (Filename.ends_with(".opt.yaml") || Filename.ends_with(".opt.yml")) {
    Classification.category = "remarks";
    Classification.description = "Optimization remarks";
    return Classification;
  }

  // Preprocessed files
  if (Extension == ".i" || Extension == ".ii") {
    Classification.category = "preprocessed";
    Classification.description = "Preprocessed source";
    return Classification;
  }

  // AST dumps
  if (Extension == ".ast" || Filename.contains("ast-dump")) {
    Classification.category = "ast";
    Classification.description = "AST dump";
    return Classification;
  }

  // Profile data
  if (Extension == ".profraw" || Extension == ".profdata") {
    Classification.category = "profile";
    Classification.description = "Profile data";
    return Classification;
  }

  // Include trees
  if (Filename.contains(".include.") || Filename.contains("include-tree")) {
    Classification.category = "include-tree";
    Classification.description = "Include tree";
    return Classification;
  }

  // Debug info
  if (Filename.contains("debug") || Filename.contains("dwarf")) {
    Classification.category = "debug";
    Classification.description = "Debug information";
    return Classification;
  }

  // Static analyzer output
  if (Filename.contains("analysis") || Filename.contains("analyzer")) {
    Classification.category = "static-analyzer";
    Classification.description = "Static analyzer output";
    return Classification;
  }

  // Macro expansion
  if (Filename.contains("macro-expanded")) {
    Classification.category = "macro-expansion";
    Classification.description = "Macro expansion";
    return Classification;
  }

  // Compilation phases
  if (Filename.contains("phases")) {
    Classification.category = "compilation-phases";
    Classification.description = "Compilation phases";
    return Classification;
  }

  // Control flow graph
  if (Extension == ".dot" || Filename.contains("cfg")) {
    Classification.category = "cfg";
    Classification.description = "Control flow graph";
    return Classification;
  }

  // Template instantiation
  if (Filename.contains("template") || Filename.contains("instantiation")) {
    Classification.category = "template-instantiation";
    Classification.description = "Template instantiation";
    return Classification;
  }

  // Default for unknown files
  Classification.category = "unknown";
  Classification.description = "Unknown file type";
  Classification.isGenerated = false;
  return Classification;
}

bool FileClassifier::shouldCollect(llvm::StringRef FilePath) const {
  auto Classification = classifyFile(FilePath);
  return Classification.category != "unknown" && Classification.isGenerated &&
         !Classification.isTemporary;
}

std::string FileClassifier::getLanguage(llvm::StringRef FilePath) const {
  StringRef Extension = sys::path::extension(FilePath);

  if (Extension == ".c")
    return "C";
  if (Extension == ".cpp" || Extension == ".cc" || Extension == ".cxx" ||
      Extension == ".C")
    return "C++";
  if (Extension == ".h" || Extension == ".hpp" || Extension == ".hh" ||
      Extension == ".hxx")
    return "Header";

  return "Unknown";
}

} // namespace advisor
} // namespace llvm
