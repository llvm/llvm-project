//===--- StaticAnalysisAnalyzer.cpp - LLVM Advisor -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs the clang static analyzer by replaying the unit's compiler invocation
// with -analyze and parsing the emitted text diagnostics.
//
//===----------------------------------------------------------------------===//

#include "Analysis/Clang/StaticAnalysisAnalyzer.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"
#include "Utils/Normalization.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
StaticAnalysisAnalyzer::run(const CapabilityContext &Context) {
  Expected<SmallVector<std::string, 32>> BaseArgsOrErr =
      buildBaseClangArgs(Context);
  if (!BaseArgsOrErr)
    return BaseArgsOrErr.takeError();

  if (Context.Unit.Arguments.empty())
    return createStringError(inconvertibleErrorCode(),
                             "missing compiler arguments");

  ErrorOr<std::string> Program =
      sys::findProgramByName(Context.Unit.Arguments.front());
  if (!Program)
    return createStringError(Program.getError(), "cannot find program: %s",
                             Context.Unit.Arguments.front().c_str());

  SmallVector<std::string, 32> Args = std::move(*BaseArgsOrErr);
  Args.push_back("-analyze");
  Args.push_back(Context.SourcePath);

  SmallString<128> StdoutPath, StderrPath;
  if (auto EC =
          sys::fs::createTemporaryFile("advisor-sa-out", "tmp", StdoutPath))
    return createStringError(EC, "failed to create stdout temp file");
  if (auto EC =
          sys::fs::createTemporaryFile("advisor-sa-err", "tmp", StderrPath))
    return createStringError(EC, "failed to create stderr temp file");

  SmallVector<StringRef, 32> ExecArgs;
  ExecArgs.push_back(*Program);
  for (const std::string &Arg : Args)
    ExecArgs.push_back(Arg);

  std::optional<StringRef> Redirects[] = {std::nullopt, StringRef(StdoutPath),
                                          StringRef(StderrPath)};

  SmallString<256> OriginalDir;
  if (!Context.WorkingDirectory.empty()) {
    sys::fs::current_path(OriginalDir);
    if (auto EC = sys::fs::set_current_path(Context.WorkingDirectory))
      return createStringError(EC, "failed to change working directory");
  }
  (void)sys::ExecuteAndWait(*Program, ExecArgs, std::nullopt, Redirects);
  if (!Context.WorkingDirectory.empty())
    (void)sys::fs::set_current_path(OriginalDir);

  std::string Stderr;
  if (auto Buf = MemoryBuffer::getFile(StderrPath))
    Stderr = (*Buf)->getBuffer().str();
  sys::fs::remove(StdoutPath);
  sys::fs::remove(StderrPath);

  // Parse clang text diagnostics: file:line:column: level: message [checker]
  Regex DiagRE("^(.+):([0-9]+):([0-9]+): ([^:]+): (.+)$", Regex::Newline);

  json::Array Findings;
  int64_t Errors = 0;
  int64_t Warnings = 0;
  int64_t Notes = 0;

  SmallVector<StringRef, 16> Lines;
  StringRef(Stderr).split(Lines, '\n');
  for (StringRef Line : Lines) {
    Line = Line.trim();
    if (Line.empty())
      continue;

    SmallVector<StringRef, 5> Matches;
    if (!DiagRE.match(Line, &Matches))
      continue;

    StringRef LevelStr = Matches[4].trim();
    StringRef LevelOut = "remark";
    if (LevelStr == "error" || LevelStr == "fatal error") {
      LevelOut = "error";
      ++Errors;
    } else if (LevelStr == "warning") {
      LevelOut = "warning";
      ++Warnings;
    } else if (LevelStr == "note") {
      LevelOut = "note";
      ++Notes;
    }

    int64_t LineNum = 0, ColNum = 0;
    Matches[2].getAsInteger(10, LineNum);
    Matches[3].getAsInteger(10, ColNum);

    json::Object Item;
    Item["level"] = LevelOut;
    Item["message"] = Matches[5].trim();
    Item["file"] = normalizePath(Matches[1], Context.WorkingDirectory);
    Item["line"] = LineNum;
    Item["column"] = ColNum;
    Findings.push_back(std::move(Item));
  }

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"errors", Errors},
      {"warnings", Warnings},
      {"notes", Notes},
      {"findings", std::move(Findings)},
  });
}
