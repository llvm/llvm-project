//===------------------- BuildExecutor.h - LLVM Advisor -------------------===//
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

#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_BUILDEXECUTOR_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_BUILDEXECUTOR_H

#include "../Config/AdvisorConfig.h"
#include "BuildContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <string>

namespace clang {
class DiagnosticConsumer;
class DiagnosticsEngine;
namespace driver {
class Driver;
class Compilation;
} // namespace driver
} // namespace clang

namespace llvm::advisor {

class BuildExecutor {
public:
  BuildExecutor(const AdvisorConfig &config);

  struct PreparedBuild {
    std::string CompilerPath;
    llvm::SmallVector<std::string, 16> InstrumentedArgs;
    std::unique_ptr<clang::driver::Driver> Driver;
    std::unique_ptr<clang::driver::Compilation> Compilation;
    std::shared_ptr<clang::DiagnosticsEngine> Diagnostics;
    std::unique_ptr<clang::DiagnosticConsumer> DiagnosticClient;
    bool UsesDriver = false;
  };

  auto execute(llvm::StringRef Compiler,
               const llvm::SmallVectorImpl<std::string> &Args,
               BuildContext &BuildCtx, llvm::StringRef TempDir,
               llvm::StringRef ArtifactRoot = llvm::StringRef())
      -> llvm::Expected<int>;

  /// Build a Compilation from the original, non-instrumented user args.
  /// The result is used exclusively for analysis (phase refinement, unit
  /// detection) and must NOT be executed.  Returns a PreparedBuild whose
  /// InstrumentedArgs are a verbatim copy of Args.
  auto buildOriginalCompilation(llvm::StringRef Compiler,
                                const llvm::SmallVectorImpl<std::string> &Args)
      -> llvm::Expected<PreparedBuild>;

  auto prepareBuild(llvm::StringRef Compiler,
                    const llvm::SmallVectorImpl<std::string> &Args,
                    BuildContext &BuildCtx, llvm::StringRef TempDir,
                    llvm::StringRef ArtifactRoot = llvm::StringRef())
      -> llvm::Expected<PreparedBuild>;

  auto executePreparedBuild(PreparedBuild &Build) -> llvm::Expected<int>;

private:
  auto instrumentCompilerArgs(llvm::StringRef CompilerPath,
                              const llvm::SmallVectorImpl<std::string> &Args,
                              BuildContext &BuildCtx, llvm::StringRef TempDir,
                              llvm::StringRef ArtifactRoot)
      -> llvm::Expected<llvm::SmallVector<std::string, 16>>;

  const AdvisorConfig &config;
};

} // namespace llvm::advisor

#endif // LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_BUILDEXECUTOR_H
