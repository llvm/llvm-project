//===--- ToolAnalyzer.cpp - LLVM Advisor --------------------------------===//
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

#include "Analysis/ToolAnalyzer.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace llvm::advisor;

ToolAnalyzer::ToolAnalyzer(StringRef CapID, StringRef ToolName,
                           ToolInputKind Kind,
                           ArrayRef<StringRef> BaseArguments, StringRef Sum)
    : CapabilityID(CapID.str()), Tool(ToolName.str()), InputKind(Kind),
      Summary(Sum.str()) {
  for (StringRef Arg : BaseArguments)
    BaseArgs.push_back(Arg.str());
}

std::string ToolAnalyzer::selectInput(const CapabilityContext &Context) const {
  switch (InputKind) {
  case ToolInputKind::None:
    return {};
  case ToolInputKind::Source:
    return Context.SourcePath;
  case ToolInputKind::Object:
    return Context.ObjectPath;
  case ToolInputKind::IR:
    return Context.IRPath;
  }
  llvm_unreachable("Unhandled ToolInputKind");
}

std::unique_ptr<CapabilityResult>
ToolAnalyzer::makeUnavailable(const CapabilityContext &Context,
                              StringRef Reason) const {
  return makeUnavailableResult(CapabilityID, Context.Unit.ID, Reason, Summary);
}

Expected<ToolInvocation>
ToolAnalyzer::buildInvocation(const CapabilityContext &Context) const {
  ToolInvocation Invocation;
  Invocation.Program = Tool;
  Invocation.Arguments.append(BaseArgs.begin(), BaseArgs.end());
  Invocation.Input = selectInput(Context);
  if (!Invocation.Input.empty())
    Invocation.Arguments.push_back(Invocation.Input);
  return Invocation;
}

Expected<json::Value>
ToolAnalyzer::processOutput(const CapabilityContext &Context,
                            const ToolInvocation &Invocation,
                            const ProcessResult &Result) const {
  json::Object Out;
  Out["capability"] = CapabilityID;
  Out["unit_id"] = Context.Unit.ID;
  Out["tool"] = Invocation.Program;
  Out["available"] = Result.ExitCode == 0;
  Out["exit_code"] = Result.ExitCode;
  Out["wall_time_ns"] = Result.WallTimeNs;
  if (!Invocation.Input.empty())
    Out["input"] = Invocation.Input;
  Out["stdout"] = Result.Stdout;
  Out["stderr"] = Result.Stderr;
  if (Result.ExitCode != 0)
    Out["reason"] = "tool exited with non-zero status";
  return json::Value(std::move(Out));
}

Expected<std::unique_ptr<CapabilityResult>>
ToolAnalyzer::run(const CapabilityContext &Context) {
  if (Tool.empty())
    return makeUnavailable(Context, "tool not configured");

  Expected<ToolInvocation> Invocation = buildInvocation(Context);
  if (!Invocation)
    return Invocation.takeError();

  if (InputKind != ToolInputKind::None && Invocation->Input.empty())
    return makeUnavailable(Context, "required input artifact is missing");

  ProcessRunner Runner;
  SmallVector<StringRef, 8> AllowedFlags;
  for (const auto &Arg : BaseArgs)
    AllowedFlags.push_back(Arg);
  Runner.allow(Invocation->Program, AllowedFlags);

  Expected<ProcessResult> Result =
      Runner.run(Invocation->Program, Invocation->Arguments);
  if (!Result)
    return makeUnavailable(Context, toString(Result.takeError()));

  Expected<json::Value> Output = processOutput(Context, *Invocation, *Result);
  if (!Output)
    return Output.takeError();

  return std::make_unique<JSONCapabilityResult>(std::move(*Output));
}
