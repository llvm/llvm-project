//===-- LLVMToolSession.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LLVMDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"

#include <cassert>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

bool matchesToolName(StringRef RegisteredName, StringRef InvokedName) {
  StringRef Stem = sys::path::stem(InvokedName);
  auto Matches = [RegisteredName](StringRef Candidate) {
    size_t Position = Candidate.rfind_insensitive(RegisteredName);
    return Position != StringRef::npos &&
           (Position + RegisteredName.size() == Candidate.size() ||
            !llvm::isAlnum(Candidate[Position + RegisteredName.size()]));
  };
  return Matches(Stem) || Matches(sys::path::filename(InvokedName));
}

bool isMulticallName(StringRef Name) { return matchesToolName("llvm", Name); }

} // namespace

struct LLVMToolSession::Impl {
  InitLLVM Initialization;
  std::string ExecutablePath;
  std::vector<std::pair<std::string, ToolMainFn>> Tools;

  Impl(int &Argc, char **&Argv, ArrayRef<CallableTool> RegisteredTools,
       bool InstallPipeSignalExitHandler, bool NeedsPOSIXUtilitySignalHandling)
      : Initialization(Argc, Argv, InstallPipeSignalExitHandler,
                       NeedsPOSIXUtilitySignalHandling),
        ExecutablePath(Argv[0]) {
    Tools.reserve(RegisteredTools.size());
    for (const CallableTool &Tool : RegisteredTools)
      Tools.emplace_back(Tool.Name.str(), Tool.Main);
  }
};

LLVMToolSession::LLVMToolSession(int &Argc, char **&Argv,
                                 ArrayRef<CallableTool> Tools,
                                 bool InstallPipeSignalExitHandler,
                                 bool NeedsPOSIXUtilitySignalHandling) {
  assert(Argc > 0 && Argv && Argv[0] &&
         "LLVMToolSession requires a valid argv[0]");
  PImpl = std::make_unique<Impl>(Argc, Argv, Tools,
                                 InstallPipeSignalExitHandler,
                                 NeedsPOSIXUtilitySignalHandling);
}

LLVMToolSession::~LLVMToolSession() = default;

ErrorOr<CallableTool> LLVMToolSession::findTool(StringRef Name) const {
  StringRef Stem = sys::path::stem(Name);
  StringRef Filename = sys::path::filename(Name);
  for (const auto &[RegisteredName, Main] : PImpl->Tools)
    if (Stem.equals_insensitive(RegisteredName) ||
        Filename.equals_insensitive(RegisteredName))
      return CallableTool{RegisteredName, Main};

  for (const auto &[RegisteredName, Main] : PImpl->Tools)
    if (matchesToolName(RegisteredName, Name))
      return CallableTool{RegisteredName, Main};
  return make_error_code(std::errc::no_such_file_or_directory);
}

ToolContext LLVMToolSession::makeContext(StringRef InvokedName,
                                         const char *PrependArg) {
  bool NeedsPrependArg = !matchesToolName(InvokedName, PImpl->ExecutablePath);
  ToolContext Context(PImpl->ExecutablePath.c_str(), PrependArg,
                      NeedsPrependArg);
  Context.Session = this;
  return Context;
}

ErrorOr<int> LLVMToolSession::callTool(ArrayRef<const char *> Args) {
  if (Args.empty())
    return make_error_code(std::errc::invalid_argument);

  StringRef InvokedName = Args.front();
  ErrorOr<CallableTool> Tool = findTool(InvokedName);
  if (!Tool) {
    if (InvokedName != PImpl->ExecutablePath && !isMulticallName(InvokedName))
      return make_error_code(std::errc::no_such_file_or_directory);
    Args = Args.drop_front();
    if (Args.empty())
      return make_error_code(std::errc::invalid_argument);
    InvokedName = Args.front();
    Tool = findTool(InvokedName);
  }

  if (!Tool)
    return Tool.getError();

  std::string PrependArg = sys::path::stem(InvokedName).str();
  ToolContext Context = makeContext(InvokedName, PrependArg.c_str());
  SmallVector<char *, 16> MutableArgs;
  MutableArgs.reserve(Args.size() + 1);
  for (const char *Arg : Args)
    MutableArgs.push_back(const_cast<char *>(Arg));
  MutableArgs.push_back(nullptr);
  return Tool->Main(Args.size(), MutableArgs.data(), Context);
}

ErrorOr<CallableTool> ToolContext::getCallableTool(StringRef Name) const {
  if (!Session)
    return make_error_code(std::errc::operation_not_permitted);
  return Session->findTool(Name);
}

ErrorOr<int> ToolContext::callTool(ArrayRef<const char *> Args) const {
  if (!Session)
    return make_error_code(std::errc::operation_not_permitted);
  return Session->callTool(Args);
}
