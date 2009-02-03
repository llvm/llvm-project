//===-- CLIDebugger.cpp - Command Line Interface to the Debugger ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the main implementation of the Command Line Interface to
// the debugger.
//
//===----------------------------------------------------------------------===//

#include "CLIDebugger.h"
#include "CLICommand.h"
#include "llvm/Debugger/SourceFile.h"
#include "llvm/ADT/StringExtras.h"
#include <iostream>
using namespace llvm;

/// CLIDebugger constructor - This initializes the debugger to its default
/// state, and initializes the command table.
///
CLIDebugger::CLIDebugger()
  : TheProgramInfo(0), TheRuntimeInfo(0), Prompt("(llvm-db) "), ListSize(10) {
  // Initialize instance variables
  CurrentFile = 0;
  LineListedStart = 1;
  LineListedEnd = 1;
  LastCurrentFrame = 0;
  CurrentLanguage = 0;

  CLICommand *C;
  //===--------------------------------------------------------------------===//
  // Program startup and shutdown options
  //
  addCommand("file", new BuiltinCLICommand(
    "Use specified file as the program to be debugged",
    "The debugger looks in the current directory and the program $PATH for the"
    " specified LLVM program.  It then unloads the currently loaded program and"
    " loads the specified program.\n",
    &CLIDebugger::fileCommand));

  addCommand("create", new BuiltinCLICommand(
    "Start the program, halting its execution in main",
    "This command creates an instance of the current program, but stops"
    "\nexecution immediately.\n",
    &CLIDebugger::createCommand));

  addCommand("kill", new BuiltinCLICommand(
    "Kills the execution of the current program being debugged", "",
    &CLIDebugger::killCommand));

  addCommand("quit", new BuiltinCLICommand(
    "Exit the debugger", "",
    &CLIDebugger::quitCommand));

  //===--------------------------------------------------------------------===//
  // Program execution commands
  //
  addCommand("run", C = new BuiltinCLICommand(
    "Start the program running from the beginning", "",
    &CLIDebugger::runCommand));
  addCommand("r", C);

  addCommand("cont", C = new BuiltinCLICommand(
    "Continue program being debugged until the next stop point", "",
    &CLIDebugger::contCommand));
  addCommand("c", C); addCommand("fg", C);

  addCommand("step", C = new BuiltinCLICommand(
    "Step program until it reaches a new source line", "",
    &CLIDebugger::stepCommand));
  addCommand("s", C);

  addCommand("next", C = new BuiltinCLICommand(
    "Step program until it reaches a new source line, stepping over calls", "",
    &CLIDebugger::nextCommand));
  addCommand("n", C);

  addCommand("finish", new BuiltinCLICommand(
    "Execute until the selected stack frame returns",
   "Upon return, the value returned is printed and put in the value history.\n",
    &CLIDebugger::finishCommand));

  //===--------------------------------------------------------------------===//
  // Stack frame commands
  //
  addCommand("backtrace", C = new BuiltinCLICommand(
   "Print backtrace of all stack frames, or innermost COUNT frames",
   "FIXME: describe.  Takes 'n', '-n' or 'full'\n",
    &CLIDebugger::backtraceCommand));
  addCommand("bt", C);

  addCommand("up", new BuiltinCLICommand(
    "Select and print stack frame that called this one",
    "An argument says how many frames up to go.\n",
    &CLIDebugger::upCommand));

  addCommand("down", new BuiltinCLICommand(
    "Select and print stack frame called by this one",
    "An argument says how many frames down go.\n",
    &CLIDebugger::downCommand));

  addCommand("frame", C = new BuiltinCLICommand(
    "Select and print a stack frame",
 "With no argument, print the selected stack frame.  (See also 'info frame').\n"
 "An argument specifies the frame to select.\n",
    &CLIDebugger::frameCommand));
  addCommand("f", C);

  //===--------------------------------------------------------------------===//
  // Breakpoint related commands
  //
  addCommand("break", C = new BuiltinCLICommand(
   "Set breakpoint at specified line or function",
   "FIXME: describe.\n",
    &CLIDebugger::breakCommand));
  addCommand("b", C);


  //===--------------------------------------------------------------------===//
  // Miscellaneous commands
  //
  addCommand("info", new BuiltinCLICommand(
    "Generic command for showing things about the program being debugged",
    "info functions: display information about functions in the program.\ninfo"
    " source : display information about the current source file.\ninfo source"
    "s : Display source file names for the program\ninfo target : print status"
    " of inferior process\n",
    &CLIDebugger::infoCommand));

  addCommand("list", C = new BuiltinCLICommand(
    "List specified function or line",
    "FIXME: document\n",
    &CLIDebugger::listCommand));
  addCommand("l", C);

  addCommand("set", new BuiltinCLICommand(
    "Change program or debugger variable",
    "FIXME: document\n",
    &CLIDebugger::setCommand));

  addCommand("show", new BuiltinCLICommand(
    "Generic command for showing things about the debugger",
    "FIXME: document\n",
    &CLIDebugger::showCommand));

  addCommand("help", C = new BuiltinCLICommand(
    "Prints information about available commands", "",
    &CLIDebugger::helpCommand));
  addCommand("h", C);
}


/// addCommand - Add a command to the CommandTable, potentially displacing a
/// preexisting command.
void CLIDebugger::addCommand(const std::string &Option, CLICommand *Cmd) {
  assert(Cmd && "Cannot set a null command!");
  CLICommand *&CS = CommandTable[Option];
  if (CS == Cmd) return; // noop

  // If we already have a command, decrement the command's reference count.
  if (CS) {
    CS->removeOptionName(Option);
    CS->dropRef();
  }
  CS = Cmd;

  // Remember that we are using this command.
  Cmd->addRef();
  Cmd->addOptionName(Option);
}

static bool isValidPrefix(const std::string &Prefix, const std::string &Option){
  return Prefix.size() <= Option.size() &&
         Prefix == std::string(Option.begin(), Option.begin()+Prefix.size());
}

/// getCommand - This looks up the specified command using a fuzzy match.
/// If the string exactly matches a command or is an unambiguous prefix of a
/// command, it returns the command.  Otherwise it throws an exception
/// indicating the possible ambiguous choices.
CLICommand *CLIDebugger::getCommand(const std::string &Command) {

  // Look up the command in the table.
  std::map<std::string, CLICommand*>::iterator CI =
    CommandTable.lower_bound(Command);

  if (Command == "") {
    throw "Null command should not get here!";
  } else if (CI == CommandTable.end() ||
             !isValidPrefix(Command, CI->first)) {
    // If this command has no relation to anything in the command table,
    // print the error message.
    throw "Unknown command: '" + Command +
          "'.  Use 'help' for list of commands.";
  } else if (CI->first == Command) {
    // We have an exact match on the command
    return CI->second;
  } else {
    // Otherwise, we have a prefix match.  Check to see if this is
    // unambiguous, and if so, run it.
    std::map<std::string, CLICommand*>::iterator CI2 = CI;

    // If the next command is a valid completion of this one, we are
    // ambiguous.
    if (++CI2 != CommandTable.end() && isValidPrefix(Command, CI2->first)) {
      std::string ErrorMsg =
        "Ambiguous command '" + Command + "'.  Options: " + CI->first;
      for (++CI; CI != CommandTable.end() &&
             isValidPrefix(Command, CI->first); ++CI)
        ErrorMsg += ", " + CI->first;
      throw ErrorMsg;
    } else {
      // It's an unambiguous prefix of a command, use it.
      return CI->second;
    }
  }
}


/// run - Start the debugger, returning when the user exits the debugger.  This
/// starts the main event loop of the CLI debugger.
///
int CLIDebugger::run() {
  std::string Command;
  std::cout << Prompt;

  // Keep track of the last command issued, so that we can reissue it if the
  // user hits enter as the command.
  CLICommand *LastCommand = 0;
  std::string LastArgs;

  // Continue reading commands until the end of file.
  while (getline(std::cin, Command)) {
    std::string Arguments = Command;

    // Split off the command from the arguments to the command.
    Command = getToken(Arguments, " \t\n\v\f\r\\/;.*&");

    try {
      CLICommand *CurCommand;

      if (Command == "") {
        CurCommand = LastCommand;
        Arguments = LastArgs;
      } else {
        CurCommand = getCommand(Command);
      }

      // Save the command we are running in case the user wants us to repeat it
      // next time.
      LastCommand = CurCommand;
      LastArgs = Arguments;

      // Finally, execute the command.
      if (CurCommand)
        CurCommand->runCommand(*this, Arguments);

    } catch (int RetVal) {
      // The quit command exits the command loop by throwing an integer return
      // code.
      return RetVal;
    } catch (const std::string &Error) {
      std::cout << "Error: " << Error << "\n";
    } catch (const char *Error) {
      std::cout << "Error: " << Error << "\n";
    } catch (const NonErrorException &E) {
      std::cout << E.getMessage() << "\n";
    } catch (...) {
      std::cout << "ERROR: Debugger caught unexpected exception!\n";
      // Attempt to continue.
    }

    // Write the prompt to get the next bit of user input
    std::cout << Prompt;
  }

  return 0;
}


/// askYesNo - Ask the user a question, and demand a yes/no response.  If
/// the user says yes, return true.
///
bool CLIDebugger::askYesNo(const std::string &Message) const {
  std::string Answer;
  std::cout << Message << " (y or n) " << std::flush;
  while (getline(std::cin, Answer)) {
    std::string Val = getToken(Answer);
    if (getToken(Answer).empty()) {
      if (Val == "yes" || Val == "y" || Val == "YES" || Val == "Y" ||
          Val == "Yes")
        return true;
      if (Val == "no" || Val == "n" || Val == "NO" || Val == "N" ||
          Val == "No")
        return false;
    }

    std::cout << "Please answer y or n.\n" << Message << " (y or n) "
              << std::flush;
  }

  // Ran out of input?
  return false;
}
