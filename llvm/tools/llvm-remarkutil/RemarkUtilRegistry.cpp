//===- RemarkUtilRegistry.cpp: Implement a command registry. --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement a simple subcommand registry.
//
//===----------------------------------------------------------------------===//
#include "RemarkUtilRegistry.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
namespace remarkutil {

using HandlerType = std::function<Error()>;

static DenseMap<cl::SubCommand *, HandlerType> &getCommands() {
  static DenseMap<cl::SubCommand *, HandlerType> Commands;
  return Commands;
}

CommandRegistration::CommandRegistration(cl::SubCommand *SC,
                                         HandlerType Command) {
  assert(getCommands().count(SC) == 0 &&
         "Attempting to overwrite a command handler");
  assert(Command && "Attempting to register an empty std::function<Error()>");
  getCommands()[SC] = Command;
}

HandlerType dispatch(cl::SubCommand *SC) {
  auto It = getCommands().find(SC);
  assert(It != getCommands().end() &&
         "Attempting to dispatch on un-registered SubCommand.");
  return It->second;
}

} // namespace remarkutil
} // namespace llvm
