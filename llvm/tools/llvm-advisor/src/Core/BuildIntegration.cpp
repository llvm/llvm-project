//===------------------- BuildIntegration.cpp - LLVM Advisor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ingests compile_commands.json and normalizes build commands.
// Bridges between build system and Advisor's internal representation.
//
//===----------------------------------------------------------------------===//

#include "Core/BuildIntegration.h"
#include "Utils/JSON.h"
#include "llvm/Support/Path.h"

#include <optional>

using namespace llvm;
using namespace llvm::advisor;

/// Best-effort shell-like split of a single command string.
/// Handles double-quoted segments and \" escapes.
/// Does not handle single quotes, variable expansion, or other shell
/// metacharacters. compile_commands.json consumers should prefer the
/// "arguments" array, which is unambiguous.
static SmallVector<std::string, 16> splitCommand(StringRef Command) {
  SmallVector<std::string, 16> Args;
  while (!Command.empty()) {
    Command = Command.ltrim(' ');
    if (Command.empty())
      break;

    std::string Arg;
    if (Command.front() == '"') {
      Command = Command.drop_front();
      while (!Command.empty() && Command.front() != '"') {
        if (Command.front() == '\\' && Command.size() > 1) {
          Arg.push_back(Command[1]);
          Command = Command.drop_front(2);
        } else {
          Arg.push_back(Command.front());
          Command = Command.drop_front();
        }
      }
      if (!Command.empty() && Command.front() == '"')
        Command = Command.drop_front();
    } else {
      size_t End = Command.find(' ');
      if (End == StringRef::npos) {
        Arg = Command.str();
        Command = {};
      } else {
        Arg = Command.slice(0, End).str();
        Command = Command.drop_front(End);
      }
    }
    Args.push_back(std::move(Arg));
  }
  return Args;
}

Expected<SmallVector<CompileCommand, 64>>
BuildIntegration::loadCompileCommands(StringRef BuildRoot) const {
  SmallString<256> Path(BuildRoot);
  sys::path::append(Path, "compile_commands.json");

  Expected<json::Value> Parsed = parseJSONFile(Path);
  if (!Parsed)
    return Parsed.takeError();
  const json::Array *Array = Parsed->getAsArray();
  if (!Array)
    return createStringError(inconvertibleErrorCode(),
                             "compile_commands.json is not an array");

  SmallVector<CompileCommand, 64> Commands;
  for (const json::Value &Value : *Array) {
    const json::Object *Object = Value.getAsObject();
    if (!Object)
      continue;
    CompileCommand Command;
    if (std::optional<StringRef> Directory = Object->getString("directory"))
      Command.Directory = Directory->str();
    if (std::optional<StringRef> File = Object->getString("file"))
      Command.File = File->str();
    if (const json::Array *Arguments = Object->getArray("arguments")) {
      for (const json::Value &Arg : *Arguments) {
        if (std::optional<StringRef> S = Arg.getAsString())
          Command.Arguments.push_back(S->str());
      }
    }
    if (Command.Arguments.empty()) {
      if (std::optional<StringRef> Line = Object->getString("command"))
        Command.Arguments = splitCommand(*Line);
    }
    if (!Command.File.empty())
      Commands.push_back(std::move(Command));
  }
  return Commands;
}
