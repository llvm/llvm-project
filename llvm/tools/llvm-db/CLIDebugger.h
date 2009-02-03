//===- CLIDebugger.h - LLVM Command Line Interface Debugger -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the CLIDebugger class, which implements a command line
// interface to the LLVM Debugger library.
//
//===----------------------------------------------------------------------===//

#ifndef CLIDEBUGGER_H
#define CLIDEBUGGER_H

#include "llvm/Debugger/Debugger.h"
#include <map>

namespace llvm {
  class CLICommand;
  class SourceFile;
  class SourceLanguage;
  class ProgramInfo;
  class RuntimeInfo;

  /// CLIDebugger - This class implements the command line interface for the
  /// LLVM debugger.
  class CLIDebugger {
    /// Dbg - The low-level LLVM debugger object that we use to do our dirty
    /// work.
    Debugger Dbg;

    /// CommandTable - This table contains a mapping from command names to the
    /// CLICommand object that implements the command.
    std::map<std::string, CLICommand*> CommandTable;

    //===------------------------------------------------------------------===//
    // Data related to the program that is currently loaded.  Note that the Dbg
    // variable also captures some information about the loaded program.  This
    // pointer is non-null iff Dbg.isProgramLoaded() is true.
    //
    ProgramInfo *TheProgramInfo;

    //===------------------------------------------------------------------===//
    // Data related to the program that is currently executing, but has stopped.
    // Note that the Dbg variable also captures some information about the
    // loaded program.  This pointer is non-null iff Dbg.isProgramRunning() is
    // true.
    //
    RuntimeInfo *TheRuntimeInfo;

    /// LastCurrentFrame - This variable holds the Frame ID of the top-level
    /// stack frame from the last time that the program was executed.  We keep
    /// this because we only want to print the source location when the current
    /// function changes.
    void *LastCurrentFrame;

    //===------------------------------------------------------------------===//
    // Data directly exposed through the debugger prompt
    //
    std::string Prompt;   // set prompt, show prompt
    unsigned ListSize;    // set listsize, show listsize

    //===------------------------------------------------------------------===//
    // Data to support user interaction
    //

    /// CurrentFile - The current source file we are inspecting, or null if
    /// none.
    const SourceFile *CurrentFile;
    unsigned LineListedStart, LineListedEnd;

    /// CurrentLanguage - This contains the source language in use, if one is
    /// explicitly set by the user.  If this is null (the default), the language
    /// is automatically determined from the current stack frame.
    ///
    const SourceLanguage *CurrentLanguage;

  public:
    CLIDebugger();

    /// getDebugger - Return the current LLVM debugger implementation being
    /// used.
    Debugger &getDebugger() { return Dbg; }

    /// run - Start the debugger, returning when the user exits the debugger.
    /// This starts the main event loop of the CLI debugger.
    ///
    int run();

    /// addCommand - Add a command to the CommandTable, potentially displacing a
    /// preexisting command.
    void addCommand(const std::string &Option, CLICommand *Cmd);

    /// addSourceDirectory - Add a directory to search when looking for the
    /// source code of the program.
    void addSourceDirectory(const std::string &Dir) {
      // FIXME: implement
    }

    /// getCurrentLanguage - Return the current source language that the user is
    /// playing around with.  This is aquired from the current stack frame of a
    /// running program if one exists, but this value can be explicitly set by
    /// the user as well.
    const SourceLanguage &getCurrentLanguage() const;

    /// getProgramInfo - Return a reference to the ProgramInfo object for the
    /// currently loaded program.  If there is no program loaded, throw an
    /// exception.
    ProgramInfo &getProgramInfo() const {
      if (TheProgramInfo == 0)
        throw "No program is loaded.";
      return *TheProgramInfo;
    }

    /// getRuntimeInfo - Return a reference to the current RuntimeInfo object.
    /// If there is no program running, throw an exception.
    RuntimeInfo &getRuntimeInfo() const {
      if (TheRuntimeInfo == 0)
        throw "No program is running.";
      return *TheRuntimeInfo;
    }

  private:   // Internal implementation methods

    /// getCommand - This looks up the specified command using a fuzzy match.
    /// If the string exactly matches a command or is an unambiguous prefix of a
    /// command, it returns the command.  Otherwise it throws an exception
    /// indicating the possible ambiguous choices.
    CLICommand *getCommand(const std::string &Command);

    /// askYesNo - Ask the user a question, and demand a yes/no response.  If
    /// the user says yes, return true.
    bool askYesNo(const std::string &Message) const;

    /// printProgramLocation - Given a loaded and created child process that has
    /// stopped, print its current source location.
    void printProgramLocation(bool PrintLocation = true);

    /// eliminateRunInfo - We are about to run the program.  Forget any state
    /// about how the program used to be stopped.
    void eliminateRunInfo();

    /// programStoppedSuccessfully - This method updates internal data
    /// structures to reflect the fact that the program just executed a while,
    /// and has successfully stopped.
    void programStoppedSuccessfully();

  public:   /// Builtin debugger commands, invokable by the user
    // Program startup and shutdown options
    void fileCommand(std::string &Options);   // file
    void createCommand(std::string &Options); // create
    void killCommand(std::string &Options);   // kill
    void quitCommand(std::string &Options);   // quit

    // Program execution commands
    void runCommand(std::string &Options);    // run|r
    void contCommand(std::string &Options);   // cont|c|fg
    void stepCommand(std::string &Options);   // step|s [count]
    void nextCommand(std::string &Options);   // next|n [count]
    void finishCommand(std::string &Options); // finish

    // Stack frame commands
    void backtraceCommand(std::string &Options); // backtrace|bt [count]
    void upCommand(std::string &Options);        // up
    void downCommand(std::string &Options);      // down
    void frameCommand(std::string &Options);     // frame


    // Breakpoint related commands
    void breakCommand(std::string &Options);  // break|b <id>

    // Miscellaneous commands
    void infoCommand(std::string &Options);   // info
    void listCommand(std::string &Options);   // list
    void setCommand(std::string &Options);    // set
    void showCommand(std::string &Options);   // show
    void helpCommand(std::string &Options);   // help

  private:
    /// startProgramRunning - If the program has been updated, reload it, then
    /// start executing the program.
    void startProgramRunning();

    /// printSourceLine - Print the specified line of the current source file.
    /// If the specified line is invalid (the source file could not be loaded or
    /// the line number is out of range), don't print anything, but return true.
    bool printSourceLine(unsigned LineNo);

    /// parseLineSpec - Parses a line specifier, for use by the 'list' command.
    /// If SourceFile is returned as a void pointer, then it was not specified.
    /// If the line specifier is invalid, an exception is thrown.
    void parseLineSpec(std::string &LineSpec, const SourceFile *&SourceFile,
                       unsigned &LineNo);

    /// parseProgramOptions - This method parses the Options string and loads it
    /// as options to be passed to the program.  This is used by the run command
    /// and by 'set args'.
    void parseProgramOptions(std::string &Options);
  };
}

#endif
