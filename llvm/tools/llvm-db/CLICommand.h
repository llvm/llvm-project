//===- CLICommand.h - Classes used to represent commands --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a small class hierarchy used to represent the various types
// of commands in the CLI debugger front-end.
//
//===----------------------------------------------------------------------===//

#ifndef CLICOMMAND_H
#define CLICOMMAND_H

#include <string>
#include <vector>
#include <cassert>

namespace llvm {
  class CLIDebugger;

  /// CLICommand - Base class of the hierarchy, used to provide the abstract
  /// interface common to all commands.
  ///
  class CLICommand {
    /// ShortHelp, LongHelp - The short and long helps strings printed for the
    /// command.  The ShortHelp string should be a single line of text without a
    /// newline.  The LongHelp string should be a full description with
    /// terminating newline.
    std::string ShortHelp, LongHelp;

    /// RefCount - This contains the number of entries in the CLIDebugger
    /// CommandTable that points to this command.
    unsigned RefCount;

    /// OptionNames - This contains a list of names for the option.  Keeping
    /// track of this is done just to make the help output more helpful.
    ///
    std::vector<std::string> OptionNames;
  public:
    CLICommand(const std::string &SH, const std::string &LH)
      : ShortHelp(SH), LongHelp(LH), RefCount(0) {}

    virtual ~CLICommand() {}

    /// addRef/dropRef - Implement a simple reference counting scheme to make
    /// sure we delete commands that are no longer used.
    void addRef() { ++RefCount; }
    void dropRef() {
      if (--RefCount == 0) delete this;
    }

    /// getPrimaryOptionName - Return the first name the option was added under.
    /// This is the name we report for the option in the help output.
    std::string getPrimaryOptionName() const {
      return OptionNames.empty() ? "" : OptionNames[0];
    }

    /// getOptionName - Return all of the names the option is registered as.
    ///
    const std::vector<std::string> &getOptionNames() const {
      return OptionNames;
    }

    /// addOptionName - Add a name that this option is known as.
    ///
    void addOptionName(const std::string &Name) {
      OptionNames.push_back(Name);
    }

    /// removeOptionName - Eliminate one of the names for this option.
    ///
    void removeOptionName(const std::string &Name) {
      unsigned i = 0;
      for (; OptionNames[i] != Name; ++i)
        assert(i+1 < OptionNames.size() && "Didn't find option name!");
      OptionNames.erase(OptionNames.begin()+i);
    }


    /// getShortHelp - Return the short help string for this command.
    ///
    const std::string &getShortHelp() { return ShortHelp; }

    /// getLongHelp - Return the long help string for this command, if it
    /// exists.
    const std::string &getLongHelp() { return LongHelp; }

    virtual void runCommand(CLIDebugger &D, std::string &Arguments) = 0;
  };

  /// BuiltinCLICommand - This class represents commands that are built directly
  /// into the debugger.
  class BuiltinCLICommand : public CLICommand {
    // Impl - Pointer to the method that implements the command
    void (CLIDebugger::*Impl)(std::string&);
  public:
    BuiltinCLICommand(const std::string &ShortHelp, const std::string &LongHelp,
                      void (CLIDebugger::*impl)(std::string&))
      : CLICommand(ShortHelp, LongHelp), Impl(impl) {}

    void runCommand(CLIDebugger &D, std::string &Arguments) {
      (D.*Impl)(Arguments);
    }
  };
}

#endif
