//===--------------- DriverContext.cpp - Shared Clang Driver --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DriverContext.h"
#include "../Config/AdvisorConfig.h"
#include "CompilationUnit.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

namespace llvm::advisor {

DriverContext::DriverContext() = default;
DriverContext::DriverContext(DriverContext &&) noexcept = default;
auto DriverContext::operator=(DriverContext &&) noexcept
    -> DriverContext & = default;
DriverContext::~DriverContext() = default;

namespace {
std::unique_ptr<DriverContext> buildContext(StringRef CompilerPath,
                                            ArrayRef<std::string> CompileFlags,
                                            ArrayRef<SourceFile> Sources) {
  auto Ctx = std::make_unique<DriverContext>();
  std::string CompilerStorage = CompilerPath.str();
  Ctx->DiagnosticsOptions = std::make_unique<clang::DiagnosticOptions>();
  auto DiagBuffer = std::make_unique<clang::TextDiagnosticBuffer>();
  IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(new clang::DiagnosticIDs());
  Ctx->Diagnostics = std::make_unique<clang::DiagnosticsEngine>(
      DiagIDs, *Ctx->DiagnosticsOptions, DiagBuffer.get(),
      /*ShouldOwnClient=*/false);

  auto Driver = std::make_unique<clang::driver::Driver>(
      CompilerStorage, llvm::sys::getDefaultTargetTriple(), *Ctx->Diagnostics);
  Driver->setTitle("llvm-advisor");
  Driver->setCheckInputsExist(false);

  SmallVector<const char *, 64> Argv;
  Argv.push_back(CompilerStorage.c_str());
  for (const auto &Flag : CompileFlags)
    Argv.push_back(Flag.c_str());
  for (const auto &Source : Sources)
    Argv.push_back(Source.path.c_str());

  auto *Compilation = Driver->BuildCompilation(Argv);
  if (!Compilation)
    return nullptr;

  Ctx->Client = std::move(DiagBuffer);
  Ctx->Driver = std::move(Driver);
  Ctx->Compilation.reset(Compilation);
  return Ctx;
}
} // namespace

std::unique_ptr<DriverContext>
createDriverContext(const AdvisorConfig &Config,
                    const CompilationUnitInfo &UnitInfo) {
  auto Compiler = UnitInfo.compilerPath.empty() ? Config.getToolPath("clang")
                                                : UnitInfo.compilerPath;
  if (Compiler.empty())
    return nullptr;
  return buildContext(Compiler, UnitInfo.compileFlags, UnitInfo.sources);
}

const clang::driver::JobList *collectCompileJobs(const DriverContext &Ctx) {
  if (!Ctx.Compilation)
    return nullptr;
  return &Ctx.Compilation->getJobs();
}

} // namespace llvm::advisor
