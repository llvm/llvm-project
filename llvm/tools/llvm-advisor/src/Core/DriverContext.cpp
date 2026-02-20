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
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Program.h"

using namespace llvm;

namespace llvm::advisor {

namespace {
std::unique_ptr<DriverContext>
buildContext(StringRef CompilerPath,
             ArrayRef<std::string> CompileFlags,
             ArrayRef<SourceFile> Sources) {
  auto Ctx = std::make_unique<DriverContext>();
  auto DiagOpts = std::make_shared<clang::DiagnosticOptions>();
  auto DiagBuffer = std::make_unique<clang::TextDiagnosticBuffer>();
  IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs(new clang::DiagnosticIDs());
  Ctx->Diagnostics = std::make_shared<clang::DiagnosticsEngine>(
      DiagIDs, DiagOpts.get(), DiagBuffer.get());

  auto Driver = std::make_unique<clang::driver::Driver>(
      CompilerPath, llvm::sys::getDefaultTargetTriple(), *Ctx->Diagnostics);
  Driver->setTitle("llvm-advisor");
  Driver->setCheckInputsExist(false);

  SmallVector<const char *, 64> Argv;
  Argv.push_back(Driver->getExecutablePath().c_str());
  for (const auto &Flag : CompileFlags)
    Argv.push_back(Flag.c_str());
  for (const auto &Source : Sources)
    Argv.push_back(Source.path.c_str());

  auto Compilation = Driver->BuildCompilation(Argv);
  if (!Compilation)
    return nullptr;

  Ctx->Driver = std::move(Driver);
  Ctx->Compilation = std::move(Compilation);
  Ctx->Client = std::move(DiagBuffer);
  return Ctx;
}
} // namespace

std::unique_ptr<DriverContext>
createDriverContext(const AdvisorConfig &Config,
                    const CompilationUnitInfo &UnitInfo) {
  auto Compiler = Config.getToolPath("clang");
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
