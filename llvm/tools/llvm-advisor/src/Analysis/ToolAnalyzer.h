//===--- ToolAnalyzer.h - LLVM Advisor -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helper for analyzers that shell out to LLVM/Clang tools.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"
#include "Utils/ProcessRunner.h"

namespace llvm::advisor {

enum class ToolInputKind { None, Source, Object, IR };

struct ToolInvocation {
  std::string Program;
  SmallVector<std::string, 16> Arguments;
  std::string Input;
};

class ToolAnalyzer : public CapabilityRunner {
public:
  ToolAnalyzer(StringRef CapID, StringRef ToolName, ToolInputKind Kind,
               ArrayRef<StringRef> BaseArguments = {},
               StringRef Sum = {});

  StringRef getCapabilityID() const override { return CapabilityID; }
  Expected<std::unique_ptr<CapabilityResult>>
  run(const CapabilityContext &Context) override;

protected:
  virtual Expected<ToolInvocation>
  buildInvocation(const CapabilityContext &Context) const;
  virtual Expected<json::Value>
  processOutput(const CapabilityContext &Context,
                const ToolInvocation &Invocation,
                const ProcessResult &Result) const;

  std::unique_ptr<CapabilityResult>
  makeUnavailable(const CapabilityContext &Context, StringRef Reason) const;

  std::string selectInput(const CapabilityContext &Context) const;

private:
  std::string CapabilityID;
  std::string Tool;
  SmallVector<std::string, 8> BaseArgs;
  ToolInputKind InputKind;
  std::string Summary;
};

} // namespace llvm::advisor
