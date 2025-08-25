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
FileClassifier::classifyFile(llvm::StringRef filePath) const {
  StringRef filename = sys::path::filename(filePath);
  StringRef extension = sys::path::extension(filePath);

  FileClassification classification;
  classification.isGenerated = true;
  classification.isTemporary = false;

  // LLVM IR files
  if (extension == ".ll") {
    classification.category = "ir";
    classification.description = "LLVM IR text";
    return classification;
  }

  // Assembly files
  if (extension == ".s" || extension == ".S") {
    classification.category = "assembly";
    classification.description = "Assembly";
    return classification;
  }

  // Optimization remarks
  if (filename.ends_with(".opt.yaml") || filename.ends_with(".opt.yml")) {
    classification.category = "remarks";
    classification.description = "Optimization remarks";
    return classification;
  }

  // Preprocessed files
  if (extension == ".i" || extension == ".ii") {
    classification.category = "preprocessed";
    classification.description = "Preprocessed source";
    return classification;
  }

  // AST dumps
  if (extension == ".ast" || filename.contains("ast-dump")) {
    classification.category = "ast";
    classification.description = "AST dump";
    return classification;
  }

  // Profile data
  if (extension == ".profraw" || extension == ".profdata") {
    classification.category = "profile";
    classification.description = "Profile data";
    return classification;
  }

  // Include trees
  if (filename.contains(".include.") || filename.contains("include-tree")) {
    classification.category = "include-tree";
    classification.description = "Include tree";
    return classification;
  }

  // Debug info
  if (filename.contains("debug") || filename.contains("dwarf")) {
    classification.category = "debug";
    classification.description = "Debug information";
    return classification;
  }

  // Static analyzer output
  if (filename.contains("analysis") || filename.contains("analyzer")) {
    classification.category = "static-analyzer";
    classification.description = "Static analyzer output";
    return classification;
  }

  // Macro expansion
  if (filename.contains("macro-expanded")) {
    classification.category = "macro-expansion";
    classification.description = "Macro expansion";
    return classification;
  }

  // Compilation phases
  if (filename.contains("phases")) {
    classification.category = "compilation-phases";
    classification.description = "Compilation phases";
    return classification;
  }

  // Control flow graph
  if (extension == ".dot" || filename.contains("cfg")) {
    classification.category = "cfg";
    classification.description = "Control flow graph";
    return classification;
  }

  // Template instantiation
  if (filename.contains("template") || filename.contains("instantiation")) {
    classification.category = "template-instantiation";
    classification.description = "Template instantiation";
    return classification;
  }

  // Default for unknown files
  classification.category = "unknown";
  classification.description = "Unknown file type";
  classification.isGenerated = false;
  return classification;
}

bool FileClassifier::shouldCollect(llvm::StringRef filePath) const {
  auto classification = classifyFile(filePath);
  return classification.category != "unknown" && classification.isGenerated &&
         !classification.isTemporary;
}

std::string FileClassifier::getLanguage(llvm::StringRef filePath) const {
  StringRef extension = sys::path::extension(filePath);

  if (extension == ".c")
    return "C";
  if (extension == ".cpp" || extension == ".cc" || extension == ".cxx" ||
      extension == ".C")
    return "C++";
  if (extension == ".h" || extension == ".hpp" || extension == ".hh" ||
      extension == ".hxx")
    return "Header";

  return "Unknown";
}

} // namespace advisor
} // namespace llvm
