//===--- Action.h - The LLVM Compiler Driver --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Action - encapsulates a single shell command.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMC2_ACTION_H
#define LLVM_TOOLS_LLVMC2_ACTION_H

#include <string>
#include <vector>

namespace llvmc {

  typedef std::vector<std::string> StrVector;

  /// Action - A class that encapsulates a single shell command.
  class Action {
    /// Command_ - The actual command (for example, 'ls').
    std::string Command_;
    /// Args_ - Command arguments. Stdout redirection ("> file") is allowed.
    std::vector<std::string> Args_;
    /// StopCompilation_ - Should we stop compilation after executing
    /// this action?
    bool StopCompilation_;
    /// OutFile_ - The output file name.
    std::string OutFile_;

  public:
    Action (const std::string& C, const StrVector& A,
            bool S, const std::string& O)
      : Command_(C), Args_(A), StopCompilation_(S), OutFile_(O)
    {}

    /// Execute - Executes the represented action.
    int Execute () const;
    bool StopCompilation () const { return StopCompilation_; }
    const std::string& OutFile() { return OutFile_; }
  };

}

#endif // LLVM_TOOLS_LLVMC2_ACTION_H
