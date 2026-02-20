//===---------------- DriverContext.h - Shared Clang Driver ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_DRIVERCONTEXT_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_DRIVERCONTEXT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <string>

namespace clang {
namespace driver {
class Driver;
class Compilation;
} // namespace driver
class DiagnosticsEngine;
class DiagnosticConsumer;
} // namespace clang

namespace llvm::advisor {

class AdvisorConfig;
struct CompilationUnitInfo;

struct DriverContext {
  std::unique_ptr<clang::driver::Driver> Driver;
  std::unique_ptr<clang::driver::Compilation> Compilation;
  std::shared_ptr<clang::DiagnosticsEngine> Diagnostics;
  std::unique_ptr<clang::DiagnosticConsumer> Client;
};

std::unique_ptr<DriverContext>
createDriverContext(const AdvisorConfig &Config,
                    const CompilationUnitInfo &UnitInfo);

const clang::driver::JobList *collectCompileJobs(const DriverContext &Ctx);

} // namespace llvm::advisor

#endif
