//===---------------- BuildExecutor.cpp - LLVM Advisor --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the BuildExecutor code generator driver. It provides a convenient
// command-line interface for generating an assembly file or a relocatable file,
// given LLVM bitcode.
//
//===----------------------------------------------------------------------===//

#include "BuildExecutor.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace advisor {

BuildExecutor::BuildExecutor(const AdvisorConfig &config) : config(config) {}

llvm::Expected<int>
BuildExecutor::execute(llvm::StringRef compiler,
                       const llvm::SmallVectorImpl<std::string> &args,
                       BuildContext &buildContext, llvm::StringRef tempDir) {
  auto instrumentedArgs = instrumentCompilerArgs(args, buildContext, tempDir);

  auto compilerPath = llvm::sys::findProgramByName(compiler);
  if (!compilerPath)
    return llvm::createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "Compiler not found: " + compiler.str());

  llvm::SmallVector<llvm::StringRef, 16> execArgs;
  execArgs.push_back(compiler);
  for (const auto &arg : instrumentedArgs) {
    execArgs.push_back(arg);
  }

  if (config.getVerbose()) {
    llvm::outs() << "Executing: " << compiler;
    for (const auto &arg : instrumentedArgs) {
      llvm::outs() << " " << arg;
    }
    llvm::outs() << "\n";
  }

  return llvm::sys::ExecuteAndWait(*compilerPath, execArgs);
}

// NOTE: We currently parse relevant compiler flags manually. We should prefer
// using the llvm::opt::ArgList / OptTable infrastructure (generated via
// TableGen) to parse and manipulate compiler arguments reliably. See
// clang/tools/clang-linker-wrapper/LinkerWrapperOpts.td for an example.
llvm::SmallVector<std::string, 16> BuildExecutor::instrumentCompilerArgs(
    const llvm::SmallVectorImpl<std::string> &args, BuildContext &buildContext,
    llvm::StringRef tempDir) {

  llvm::SmallVector<std::string, 16> result(args.begin(), args.end());
  llvm::DenseSet<llvm::StringRef> existingFlags;

  // Scan existing flags to avoid duplication
  for (const auto &arg : args) {
    if (llvm::StringRef(arg).starts_with("-g"))
      existingFlags.insert("debug");
    if (llvm::StringRef(arg).contains("-fsave-optimization-record"))
      existingFlags.insert("remarks");
    if (llvm::StringRef(arg).contains("-fprofile-instr-generate"))
      existingFlags.insert("profile");
  }

  // Add debug info if not present
  if (!existingFlags.contains("debug"))
    result.push_back("-g");

  // Add optimization remarks with proper redirection
  if (!existingFlags.contains("remarks")) {
    result.push_back("-fsave-optimization-record");
    result.push_back("-foptimization-record-file=" + tempDir.str() +
                     "/remarks.opt.yaml");
    buildContext.expectedGeneratedFiles.push_back(tempDir.str() +
                                                 "/remarks.opt.yaml");
  } else {
    // If user already specified remarks, find and redirect the file
    bool foundFileFlag = false;
    for (auto &arg : result) {
      if (llvm::StringRef(arg).contains("-foptimization-record-file=")) {
        // Extract filename and redirect to temp
        llvm::StringRef existingPath = llvm::StringRef(arg).substr(26);
        llvm::StringRef filename = llvm::sys::path::filename(existingPath);
        arg = "-foptimization-record-file=" + tempDir.str() + "/" +
              filename.str();
        buildContext.expectedGeneratedFiles.push_back(tempDir.str() + "/" +
                                                     filename.str());
        foundFileFlag = true;
        break;
      }
    }
    // If no explicit file specified, add our own
    if (!foundFileFlag) {
      result.push_back("-foptimization-record-file=" + tempDir.str() +
                       "/remarks.opt.yaml");
      buildContext.expectedGeneratedFiles.push_back(tempDir.str() +
                                                   "/remarks.opt.yaml");
    }
  }

  // Add profiling if enabled and not present, redirect to temp directory
  if (config.getRunProfiler() && !existingFlags.contains("profile")) {
    result.push_back("-fprofile-instr-generate=" + tempDir.str() +
                     "/profile.profraw");
    result.push_back("-fcoverage-mapping");
    buildContext.expectedGeneratedFiles.push_back(tempDir.str() +
                                                 "/profile.profraw");
  }

  // Add remark extraction flags if none present
  bool hasRpass = false;
  for (const auto &arg : result) {
    if (llvm::StringRef(arg).starts_with("-Rpass=")) {
      hasRpass = true;
      break;
    }
  }
  if (!hasRpass) {
    // For now we add offloading and general analysis passes
    result.push_back("-Rpass=kernel-info");
    result.push_back("-Rpass=analysis");
  }

  // Add diagnostic output format for better parsing
  bool hasDiagFormat = false;
  for (const auto &arg : result) {
    if (llvm::StringRef(arg).contains("-fdiagnostics-format")) {
      hasDiagFormat = true;
      break;
    }
  }
  if (!hasDiagFormat) {
    result.push_back("-fdiagnostics-parseable-fixits");
    result.push_back("-fdiagnostics-absolute-paths");
  }

  return result;
}

} // namespace advisor
} // namespace llvm
