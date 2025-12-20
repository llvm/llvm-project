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
#include <string>

namespace llvm::advisor {

class BuildExecutor {
public:
  BuildExecutor(const AdvisorConfig &config);

  auto execute(llvm::StringRef compiler,
               const llvm::SmallVectorImpl<std::string> &args,
               BuildContext &buildContext, llvm::StringRef tempDir)
      -> llvm::Expected<int>;

private:
  auto instrumentCompilerArgs(const llvm::SmallVectorImpl<std::string> &args,
                              BuildContext &buildContext,
                              llvm::StringRef tempDir)
      -> llvm::SmallVector<std::string, 16>;

  const AdvisorConfig &config;
};

} // namespace llvm::advisor

#endif // LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_BUILDEXECUTOR_H
