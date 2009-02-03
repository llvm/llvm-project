//===-- Commands.cpp - Implement various commands for the CLI -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements many builtin user commands.
//
//===----------------------------------------------------------------------===//

#include "CLIDebugger.h"
#include "CLICommand.h"
#include "llvm/Debugger/ProgramInfo.h"
#include "llvm/Debugger/RuntimeInfo.h"
#include "llvm/Debugger/SourceLanguage.h"
#include "llvm/Debugger/SourceFile.h"
#include "llvm/Debugger/InferiorProcess.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/ADT/StringExtras.h"
#include <iostream>
#include <cstdlib>
using namespace llvm;

/// getCurrentLanguage - Return the current source language that the user is
/// playing around with.  This is aquired from the current stack frame of a
/// running program if one exists, but this value can be explicitly set by the
/// user as well.
const SourceLanguage &CLIDebugger::getCurrentLanguage() const {
  // If the user explicitly switched languages with 'set language', use what
  // they asked for.
  if (CurrentLanguage) {
    return *CurrentLanguage;
  } else if (Dbg.isProgramRunning()) {
    // Otherwise, if the program is running, infer the current language from it.
    const GlobalVariable *FuncDesc =
      getRuntimeInfo().getCurrentFrame().getFunctionDesc();
    return getProgramInfo().getFunction(FuncDesc).getSourceFile().getLanguage();
  } else {
    // Otherwise, default to C like GDB apparently does.
    return SourceLanguage::getCFamilyInstance();
  }
}

/// startProgramRunning - If the program has been updated, reload it, then
/// start executing the program.
void CLIDebugger::startProgramRunning() {
  eliminateRunInfo();

  // If the program has been modified, reload it!
  sys::PathWithStatus Program(Dbg.getProgramPath());
  std::string Err;
  const sys::FileStatus *Status = Program.getFileStatus(false, &Err);
  if (!Status)
    throw Err;
  if (TheProgramInfo->getProgramTimeStamp() != Status->getTimestamp()) {
    std::cout << "'" << Program << "' has changed; re-reading program.\n";

    // Unload an existing program.  This kills the program if necessary.
    Dbg.unloadProgram();
    delete TheProgramInfo;
    TheProgramInfo = 0;
    CurrentFile = 0;

    Dbg.loadProgram(Program.toString());
    TheProgramInfo = new ProgramInfo(Dbg.getProgram());
  }

  std::cout << "Starting program: " << Dbg.getProgramPath() << "\n";
  Dbg.createProgram();

  // There was no current frame.
  LastCurrentFrame = 0;
}

/// printSourceLine - Print the specified line of the current source file.
/// If the specified line is invalid (the source file could not be loaded or
/// the line number is out of range), don't print anything, but return true.
bool CLIDebugger::printSourceLine(unsigned LineNo) {
  assert(CurrentFile && "There is no current source file to print!");
  const char *LineStart, *LineEnd;
  CurrentFile->getSourceLine(LineNo-1, LineStart, LineEnd);
  if (LineStart == 0) return true;
  std::cout << LineNo;

  // If this is the line the program is currently stopped at, print a marker.
  if (Dbg.isProgramRunning()) {
    unsigned CurLineNo, CurColNo;
    const SourceFileInfo *CurSFI;
    getRuntimeInfo().getCurrentFrame().getSourceLocation(CurLineNo, CurColNo,
                                                         CurSFI);

    if (CurLineNo == LineNo && CurrentFile == &CurSFI->getSourceText())
      std::cout << " ->";
  }

  std::cout << "\t" << std::string(LineStart, LineEnd) << "\n";
  return false;
}

/// printProgramLocation - Print a line of the place where the current stack
/// frame has stopped and the source line it is on.
///
void CLIDebugger::printProgramLocation(bool PrintLocation) {
  assert(Dbg.isProgramLoaded() && Dbg.isProgramRunning() &&
         "Error program is not loaded and running!");

  // Figure out where the program stopped...
  StackFrame &SF = getRuntimeInfo().getCurrentFrame();
  unsigned LineNo, ColNo;
  const SourceFileInfo *FileDesc;
  SF.getSourceLocation(LineNo, ColNo, FileDesc);

  // If requested, print out some program information about WHERE we are.
  if (PrintLocation) {
    // FIXME: print the current function arguments
    if (const GlobalVariable *FuncDesc = SF.getFunctionDesc())
      std::cout << getProgramInfo().getFunction(FuncDesc).getSymbolicName();
    else
      std::cout << "<unknown function>";

    CurrentFile = &FileDesc->getSourceText();

    std::cout << " at " << CurrentFile->getFilename() << ":" << LineNo;
    if (ColNo) std::cout << ":" << ColNo;
    std::cout << "\n";
  }

  if (printSourceLine(LineNo))
    std::cout << "<could not load source file>\n";
  else {
    LineListedStart = LineNo-ListSize/2+1;
    if ((int)LineListedStart < 1) LineListedStart = 1;
    LineListedEnd = LineListedStart+1;
  }
}

/// eliminateRunInfo - We are about to run the program.  Forget any state
/// about how the program used to be stopped.
void CLIDebugger::eliminateRunInfo() {
  delete TheRuntimeInfo;
  TheRuntimeInfo = 0;
}

/// programStoppedSuccessfully - This method updates internal data
/// structures to reflect the fact that the program just executed a while,
/// and has successfully stopped.
void CLIDebugger::programStoppedSuccessfully() {
  assert(TheRuntimeInfo==0 && "Someone forgot to release the old RuntimeInfo!");

  TheRuntimeInfo = new RuntimeInfo(TheProgramInfo, Dbg.getRunningProcess());

  // FIXME: if there are any breakpoints at the current location, print them as
  // well.

  // Since the program as successfully stopped, print its location.
  void *CurrentFrame = getRuntimeInfo().getCurrentFrame().getFrameID();
  printProgramLocation(CurrentFrame != LastCurrentFrame);
  LastCurrentFrame = CurrentFrame;
}



/// getUnsignedIntegerOption - Get an unsigned integer number from the Val
/// string.  Check to make sure that the string contains an unsigned integer
/// token, and if not, throw an exception.  If isOnlyOption is set, also throw
/// an exception if there is extra junk at the end of the string.
static unsigned getUnsignedIntegerOption(const char *Msg, std::string &Val,
                                         bool isOnlyOption = true) {
  std::string Tok = getToken(Val);
  if (Tok.empty() || (isOnlyOption && !getToken(Val).empty()))
    throw std::string(Msg) + " expects an unsigned integer argument.";

  char *EndPtr;
  unsigned Result = strtoul(Tok.c_str(), &EndPtr, 0);
  if (EndPtr != Tok.c_str()+Tok.size())
    throw std::string(Msg) + " expects an unsigned integer argument.";

  return Result;
}

/// getOptionalUnsignedIntegerOption - This method is just like
/// getUnsignedIntegerOption, but if the argument value is not specified, a
/// default is returned instead of causing an error.
static unsigned
getOptionalUnsignedIntegerOption(const char *Msg, unsigned Default,
                                 std::string &Val, bool isOnlyOption = true) {
  // Check to see if the value was specified...
  std::string TokVal = getToken(Val);
  if (TokVal.empty()) return Default;

  // If it was specified, add it back to the value we are parsing...
  Val = TokVal+Val;

  // And parse normally.
  return getUnsignedIntegerOption(Msg, Val, isOnlyOption);
}


/// parseProgramOptions - This method parses the Options string and loads it
/// as options to be passed to the program.  This is used by the run command
/// and by 'set args'.
void CLIDebugger::parseProgramOptions(std::string &Options) {
  // FIXME: tokenizing by whitespace is clearly incorrect.  Instead we should
  // honor quotes and other things that a shell would.  Also in the future we
  // should support redirection of standard IO.

  std::vector<std::string> Arguments;
  for (std::string A = getToken(Options); !A.empty(); A = getToken(Options))
    Arguments.push_back(A);
  Dbg.setProgramArguments(Arguments.begin(), Arguments.end());
}


//===----------------------------------------------------------------------===//
//                   Program startup and shutdown options
//===----------------------------------------------------------------------===//


/// file command - If the user specifies an option, search the PATH for the
/// specified program/bitcode file and load it.  If the user does not specify
/// an option, unload the current program.
void CLIDebugger::fileCommand(std::string &Options) {
  std::string Prog = getToken(Options);
  if (!getToken(Options).empty())
    throw "file command takes at most one argument.";

  // Check to make sure the user knows what they are doing
  if (Dbg.isProgramRunning() &&
      !askYesNo("A program is already loaded.  Kill it?"))
    return;

  // Unload an existing program.  This kills the program if necessary.
  eliminateRunInfo();
  delete TheProgramInfo;
  TheProgramInfo = 0;
  Dbg.unloadProgram();
  CurrentFile = 0;

  // If requested, start the new program.
  if (Prog.empty()) {
    std::cout << "Unloaded program.\n";
  } else {
    std::cout << "Loading program... " << std::flush;
    Dbg.loadProgram(Prog);
    assert(Dbg.isProgramLoaded() &&
           "loadProgram succeeded, but not program loaded!");
    TheProgramInfo = new ProgramInfo(Dbg.getProgram());
    std::cout << "successfully loaded '" << Dbg.getProgramPath() << "'!\n";
  }
}


void CLIDebugger::createCommand(std::string &Options) {
  if (!getToken(Options).empty())
    throw "create command does not take any arguments.";
  if (!Dbg.isProgramLoaded()) throw "No program loaded.";
  if (Dbg.isProgramRunning() &&
      !askYesNo("The program is already running.  Restart from the beginning?"))
    return;

  // Start the program running.
  startProgramRunning();

  // The program stopped!
  programStoppedSuccessfully();
}

void CLIDebugger::killCommand(std::string &Options) {
  if (!getToken(Options).empty())
    throw "kill command does not take any arguments.";
  if (!Dbg.isProgramRunning())
    throw "No program is currently being run.";

  if (askYesNo("Kill the program being debugged?"))
    Dbg.killProgram();
  eliminateRunInfo();
}

void CLIDebugger::quitCommand(std::string &Options) {
  if (!getToken(Options).empty())
    throw "quit command does not take any arguments.";

  if (Dbg.isProgramRunning() &&
      !askYesNo("The program is running.  Exit anyway?"))
    return;

  // Throw exception to get out of the user-input loop.
  throw 0;
}


//===----------------------------------------------------------------------===//
//                        Program execution commands
//===----------------------------------------------------------------------===//

void CLIDebugger::runCommand(std::string &Options) {
  if (!Dbg.isProgramLoaded()) throw "No program loaded.";
  if (Dbg.isProgramRunning() &&
      !askYesNo("The program is already running.  Restart from the beginning?"))
    return;

  // Parse all of the options to the run command, which specify program
  // arguments to run with.
  parseProgramOptions(Options);

  eliminateRunInfo();

  // Start the program running.
  startProgramRunning();

  // Start the program running...
  Options = "";
  contCommand(Options);
}

void CLIDebugger::contCommand(std::string &Options) {
  if (!getToken(Options).empty()) throw "cont argument not supported yet.";
  if (!Dbg.isProgramRunning()) throw "Program is not running.";

  eliminateRunInfo();

  Dbg.contProgram();

  // The program stopped!
  programStoppedSuccessfully();
}

void CLIDebugger::stepCommand(std::string &Options) {
  if (!Dbg.isProgramRunning()) throw "Program is not running.";

  // Figure out how many times to step.
  unsigned Amount =
    getOptionalUnsignedIntegerOption("'step' command", 1, Options);

  eliminateRunInfo();

  // Step the specified number of times.
  for (; Amount; --Amount)
    Dbg.stepProgram();

  // The program stopped!
  programStoppedSuccessfully();
}

void CLIDebugger::nextCommand(std::string &Options) {
  if (!Dbg.isProgramRunning()) throw "Program is not running.";
  unsigned Amount =
    getOptionalUnsignedIntegerOption("'next' command", 1, Options);

  eliminateRunInfo();

  for (; Amount; --Amount)
    Dbg.nextProgram();

  // The program stopped!
  programStoppedSuccessfully();
}

void CLIDebugger::finishCommand(std::string &Options) {
  if (!getToken(Options).empty())
    throw "finish command does not take any arguments.";
  if (!Dbg.isProgramRunning()) throw "Program is not running.";

  // Figure out where we are exactly.  If the user requests that we return from
  // a frame that is not the top frame, make sure we get it.
  void *CurrentFrame = getRuntimeInfo().getCurrentFrame().getFrameID();

  eliminateRunInfo();

  Dbg.finishProgram(CurrentFrame);

  // The program stopped!
  programStoppedSuccessfully();
}

//===----------------------------------------------------------------------===//
//                           Stack frame commands
//===----------------------------------------------------------------------===//

void CLIDebugger::backtraceCommand(std::string &Options) {
  // Accepts "full", n, -n
  if (!getToken(Options).empty())
    throw "FIXME: bt command argument not implemented yet!";

  RuntimeInfo &RI = getRuntimeInfo();
  ProgramInfo &PI = getProgramInfo();

  try {
    for (unsigned i = 0; ; ++i) {
      StackFrame &SF = RI.getStackFrame(i);
      std::cout << "#" << i;
      if (i == RI.getCurrentFrameIdx())
        std::cout << " ->";
      std::cout << "\t" << SF.getFrameID() << " in ";
      if (const GlobalVariable *G = SF.getFunctionDesc())
        std::cout << PI.getFunction(G).getSymbolicName();

      unsigned LineNo, ColNo;
      const SourceFileInfo *SFI;
      SF.getSourceLocation(LineNo, ColNo, SFI);
      if (!SFI->getBaseName().empty()) {
        std::cout << " at " << SFI->getBaseName();
        if (LineNo) {
          std::cout << ":" << LineNo;
          if (ColNo)
            std::cout << ":" << ColNo;
        }
      }

      // FIXME: when we support shared libraries, we should print ' from foo.so'
      // if the stack frame is from a different object than the current one.

      std::cout << "\n";
    }
  } catch (...) {
    // Stop automatically when we run off the bottom of the stack.
  }
}

void CLIDebugger::upCommand(std::string &Options) {
  unsigned Num =
    getOptionalUnsignedIntegerOption("'up' command", 1, Options);

  RuntimeInfo &RI = getRuntimeInfo();
  unsigned CurFrame = RI.getCurrentFrameIdx();

  // Check to see if we go can up the specified number of frames.
  try {
    RI.getStackFrame(CurFrame+Num);
  } catch (...) {
    if (Num == 1)
      throw "Initial frame selected; you cannot go up.";
    else
      throw "Cannot go up " + utostr(Num) + " frames!";
  }

  RI.setCurrentFrameIdx(CurFrame+Num);
  printProgramLocation();
}

void CLIDebugger::downCommand(std::string &Options) {
  unsigned Num =
    getOptionalUnsignedIntegerOption("'down' command", 1, Options);

  RuntimeInfo &RI = getRuntimeInfo();
  unsigned CurFrame = RI.getCurrentFrameIdx();

  // Check to see if we can go up the specified number of frames.
  if (CurFrame < Num) {
    if (Num == 1)
      throw "Bottom (i.e., innermost) frame selected; you cannot go down.";
    else
      throw "Cannot go down " + utostr(Num) + " frames!";
  }

  RI.setCurrentFrameIdx(CurFrame-Num);
  printProgramLocation();
}

void CLIDebugger::frameCommand(std::string &Options) {
  RuntimeInfo &RI = getRuntimeInfo();
  unsigned CurFrame = RI.getCurrentFrameIdx();

  unsigned Num =
    getOptionalUnsignedIntegerOption("'frame' command", CurFrame, Options);

  // Check to see if we go to the specified frame.
  RI.getStackFrame(Num);

  RI.setCurrentFrameIdx(Num);
  printProgramLocation();
}


//===----------------------------------------------------------------------===//
//                        Breakpoint related commands
//===----------------------------------------------------------------------===//

void CLIDebugger::breakCommand(std::string &Options) {
  // Figure out where the user wants a breakpoint.
  const SourceFile *File;
  unsigned LineNo;

  // Check to see if the user specified a line specifier.
  std::string Option = getToken(Options);  // strip whitespace
  if (!Option.empty()) {
    Options = Option + Options;  // reconstruct string

    // Parse the line specifier.
    parseLineSpec(Options, File, LineNo);
  } else {
    // Build a line specifier for the current stack frame.
    throw "FIXME: breaking at the current location is not implemented yet!";
  }

  if (!File) File = CurrentFile;
  if (File == 0)
    throw "Unknown file to place breakpoint!";

  std::cerr << "Break: " << File->getFilename() << ":" << LineNo << "\n";

  throw "breakpoints not implemented yet!";
}

//===----------------------------------------------------------------------===//
//                          Miscellaneous commands
//===----------------------------------------------------------------------===//

void CLIDebugger::infoCommand(std::string &Options) {
  std::string What = getToken(Options);

  if (What.empty() || !getToken(Options).empty()){
    std::string infoStr("info");
    helpCommand(infoStr);
    return;
  }

  if (What == "frame") {
  } else if (What == "functions") {
    const std::map<const GlobalVariable*, SourceFunctionInfo*> &Functions
      = getProgramInfo().getSourceFunctions();
    std::cout << "All defined functions:\n";
    // FIXME: GDB groups these by source file.  We could do that I guess.
    for (std::map<const GlobalVariable*, SourceFunctionInfo*>::const_iterator
           I = Functions.begin(), E = Functions.end(); I != E; ++I) {
      std::cout << I->second->getSymbolicName() << "\n";
    }

  } else if (What == "source") {
    if (CurrentFile == 0)
      throw "No current source file.";

    // Get the SourceFile information for the current file.
    const SourceFileInfo &SF =
      getProgramInfo().getSourceFile(CurrentFile->getDescriptor());

    std::cout << "Current source file is: " << SF.getBaseName() << "\n"
              << "Compilation directory is: " << SF.getDirectory() << "\n";
    if (unsigned NL = CurrentFile->getNumLines())
      std::cout << "Located in: " << CurrentFile->getFilename() << "\n"
                << "Contains " << NL << " lines\n";
    else
      std::cout << "Could not find source file.\n";
    std::cout << "Source language is "
              << SF.getLanguage().getSourceLanguageName() << "\n";

  } else if (What == "sources") {
    const std::map<const GlobalVariable*, SourceFileInfo*> &SourceFiles =
      getProgramInfo().getSourceFiles();
    std::cout << "Source files for the program:\n";
    for (std::map<const GlobalVariable*, SourceFileInfo*>::const_iterator I =
           SourceFiles.begin(), E = SourceFiles.end(); I != E;) {
      std::cout << I->second->getDirectory() << "/"
                << I->second->getBaseName();
      ++I;
      if (I != E) std::cout << ", ";
    }
    std::cout << "\n";
  } else if (What == "target") {
    std::cout << Dbg.getRunningProcess().getStatus();
  } else {
    // See if this is something handled by the current language.
    if (getCurrentLanguage().printInfo(What))
      return;

    throw "Unknown info command '" + What + "'.  Try 'help info'.";
  }
}

/// parseLineSpec - Parses a line specifier, for use by the 'list' command.
/// If SourceFile is returned as a void pointer, then it was not specified.
/// If the line specifier is invalid, an exception is thrown.
void CLIDebugger::parseLineSpec(std::string &LineSpec,
                                const SourceFile *&SourceFile,
                                unsigned &LineNo) {
  SourceFile = 0;
  LineNo = 0;

  // First, check to see if we have a : separator.
  std::string FirstPart = getToken(LineSpec, ":");
  std::string SecondPart = getToken(LineSpec, ":");
  if (!getToken(LineSpec).empty()) throw "Malformed line specification!";

  // If there is no second part, we must have either "function", "number",
  // "+offset", or "-offset".
  if (SecondPart.empty()) {
    if (FirstPart.empty()) throw "Malformed line specification!";
    if (FirstPart[0] == '+') {
      FirstPart.erase(FirstPart.begin(), FirstPart.begin()+1);
      // For +n, return LineListedEnd+n
      LineNo = LineListedEnd +
               getUnsignedIntegerOption("Line specifier '+'", FirstPart);

    } else if (FirstPart[0] == '-') {
      FirstPart.erase(FirstPart.begin(), FirstPart.begin()+1);
      // For -n, return LineListedEnd-n
      LineNo = LineListedEnd -
               getUnsignedIntegerOption("Line specifier '-'", FirstPart);
      if ((int)LineNo < 1) LineNo = 1;
    } else if (FirstPart[0] == '*') {
      throw "Address expressions not supported as source locations!";
    } else {
      // Ok, check to see if this is just a line number.
      std::string Saved = FirstPart;
      try {
        LineNo = getUnsignedIntegerOption("", Saved);
      } catch (...) {
        // Ok, it's not a valid line number.  It must be a source-language
        // entity name.
        std::string Name = getToken(FirstPart);
        if (!getToken(FirstPart).empty())
          throw "Extra junk in line specifier after '" + Name + "'.";
        SourceFunctionInfo *SFI =
          getCurrentLanguage().lookupFunction(Name, getProgramInfo(),
                                              TheRuntimeInfo);
        if (SFI == 0)
          throw "Unknown identifier '" + Name + "'.";

        unsigned L, C;
        SFI->getSourceLocation(L, C);
        if (L == 0) throw "Could not locate '" + Name + "'!";
        LineNo = L;
        SourceFile = &SFI->getSourceFile().getSourceText();
        return;
      }
    }

  } else {
    // Ok, this must be a filename qualified line number or function name.
    // First, figure out the source filename.
    std::string SourceFilename = getToken(FirstPart);
    if (!getToken(FirstPart).empty())
      throw "Invalid filename qualified source location!";

    // Next, check to see if this is just a line number.
    std::string Saved = SecondPart;
    try {
      LineNo = getUnsignedIntegerOption("", Saved);
    } catch (...) {
      // Ok, it's not a valid line number.  It must be a function name.
      throw "FIXME: Filename qualified function names are not support "
            "as line specifiers yet!";
    }

    // Ok, we got the line number.  Now check out the source file name to make
    // sure it's all good.  If it is, return it.  If not, throw exception.
    SourceFile =&getProgramInfo().getSourceFile(SourceFilename).getSourceText();
  }
}

void CLIDebugger::listCommand(std::string &Options) {
  if (!Dbg.isProgramLoaded())
    throw "No program is loaded.  Use the 'file' command.";

  // Handle "list foo," correctly, by returning " " as the second token
  Options += " ";

  std::string FirstLineSpec = getToken(Options, ",");
  std::string SecondLineSpec = getToken(Options, ",");
  if (!getToken(Options, ",").empty())
    throw "list command only expects two source location specifiers!";

  // StartLine, EndLine - The starting and ending line numbers to print.
  unsigned StartLine = 0, EndLine = 0;

  if (SecondLineSpec.empty()) {    // No second line specifier provided?
    // Handle special forms like "", "+", "-", etc.
    std::string TmpSpec = FirstLineSpec;
    std::string Tok = getToken(TmpSpec);
    if (getToken(TmpSpec).empty() && (Tok == "" || Tok == "+" || Tok == "-")) {
      if (Tok == "+" || Tok == "") {
        StartLine = LineListedEnd;
        EndLine = StartLine + ListSize;
      } else {
        assert(Tok == "-");
        StartLine = LineListedStart-ListSize;
        EndLine = LineListedStart;
        if ((int)StartLine <= 0) StartLine = 1;
      }
    } else {
      // Must be a normal line specifier.
      const SourceFile *File;
      unsigned LineNo;
      parseLineSpec(FirstLineSpec, File, LineNo);

      // If the user only specified one file specifier, we should display
      // ListSize lines centered at the specified line.
      if (File != 0) CurrentFile = File;
      StartLine = LineNo - (ListSize+1)/2;
      if ((int)StartLine <= 0) StartLine = 1;
      EndLine = StartLine + ListSize;
    }

  } else {
    // Parse two line specifiers...
    const SourceFile *StartFile, *EndFile;
    unsigned StartLineNo, EndLineNo;
    parseLineSpec(FirstLineSpec, StartFile, StartLineNo);
    unsigned SavedLLE = LineListedEnd;
    LineListedEnd = StartLineNo;
    try {
      parseLineSpec(SecondLineSpec, EndFile, EndLineNo);
    } catch (...) {
      LineListedEnd = SavedLLE;
      throw;
    }

    // Inherit file specified by the first line spec if there was one.
    if (EndFile == 0) EndFile = StartFile;

    if (StartFile != EndFile)
      throw "Start and end line specifiers are in different files!";
    CurrentFile = StartFile;
    StartLine = StartLineNo;
    EndLine = EndLineNo+1;
  }

  assert((int)StartLine > 0 && (int)EndLine > 0 && StartLine <= EndLine &&
         "Error reading line specifiers!");

  // If there was no current file, and the user didn't specify one to list, we
  // have an error.
  if (CurrentFile == 0)
    throw "There is no current file to list.";

  // Remember for next time.
  LineListedStart = StartLine;
  LineListedEnd = StartLine;

  for (unsigned LineNo = StartLine; LineNo != EndLine; ++LineNo) {
    // Print the source line, unless it is invalid.
    if (printSourceLine(LineNo))
      break;
    LineListedEnd = LineNo+1;
  }

  // If we didn't print any lines, find out why.
  if (LineListedEnd == StartLine) {
    // See if we can read line #0 from the file, if not, we couldn't load the
    // file.
    const char *LineStart, *LineEnd;
    CurrentFile->getSourceLine(0, LineStart, LineEnd);
    if (LineStart == 0)
      throw "Could not load source file '" + CurrentFile->getFilename() + "'!";
    else
      std::cout << "<end of file>\n";
  }
}

void CLIDebugger::setCommand(std::string &Options) {
  std::string What = getToken(Options);

  if (What.empty())
    throw "set command expects at least two arguments.";
  if (What == "args") {
    parseProgramOptions(Options);
  } else if (What == "language") {
    std::string Lang = getToken(Options);
    if (!getToken(Options).empty())
      throw "set language expects one argument at most.";
    if (Lang == "") {
      std::cout << "The currently understood settings are:\n\n"
                << "local or auto  Automatic setting based on source file\n"
                << "c              Use the C language\n"
                << "c++            Use the C++ language\n"
                << "unknown        Use when source language is not supported\n";
    } else if (Lang == "local" || Lang == "auto") {
      CurrentLanguage = 0;
    } else if (Lang == "c") {
      CurrentLanguage = &SourceLanguage::getCFamilyInstance();
    } else if (Lang == "c++") {
      CurrentLanguage = &SourceLanguage::getCPlusPlusInstance();
    } else if (Lang == "unknown") {
      CurrentLanguage = &SourceLanguage::getUnknownLanguageInstance();
    } else {
      throw "Unknown language '" + Lang + "'.";
    }

  } else if (What == "listsize") {
    ListSize = getUnsignedIntegerOption("'set prompt' command", Options);
  } else if (What == "prompt") {
    // Include any trailing whitespace or other tokens, but not leading
    // whitespace.
    Prompt = getToken(Options);  // Strip leading whitespace
    Prompt += Options;           // Keep trailing whitespace or other stuff
  } else {
    // FIXME: Try to parse this as a source-language program expression.
    throw "Don't know how to set '" + What + "'!";
  }
}

void CLIDebugger::showCommand(std::string &Options) {
  std::string What = getToken(Options);

  if (What.empty() || !getToken(Options).empty())
    throw "show command expects one argument.";

  if (What == "args") {
    std::cout << "Argument list to give program when started is \"";
    // FIXME: This doesn't print stuff correctly if the arguments have spaces in
    // them, but currently the only way to get that is to use the --args command
    // line argument.  This should really handle escaping all hard characters as
    // needed.
    for (unsigned i = 0, e = Dbg.getNumProgramArguments(); i != e; ++i)
      std::cout << (i ? " " : "") << Dbg.getProgramArgument(i);
    std::cout << "\"\n";

  } else if (What == "language") {
    std::cout << "The current source language is '";
    if (CurrentLanguage)
      std::cout << CurrentLanguage->getSourceLanguageName();
    else
      std::cout << "auto; currently "
                << getCurrentLanguage().getSourceLanguageName();
    std::cout << "'.\n";
  } else if (What == "listsize") {
    std::cout << "Number of source lines llvm-db will list by default is "
              << ListSize << ".\n";
  } else if (What == "prompt") {
    std::cout << "llvm-db's prompt is \"" << Prompt << "\".\n";
  } else {
    throw "Unknown show command '" + What + "'.  Try 'help show'.";
  }
}

void CLIDebugger::helpCommand(std::string &Options) {
  // Print out all of the commands in the CommandTable
  std::string Command = getToken(Options);
  if (!getToken(Options).empty())
    throw "help command takes at most one argument.";

  // Getting detailed help on a particular command?
  if (!Command.empty()) {
    CLICommand *C = getCommand(Command);
    std::cout << C->getShortHelp() << ".\n" << C->getLongHelp();

    // If there are aliases for this option, print them out.
    const std::vector<std::string> &Names = C->getOptionNames();
    if (Names.size() > 1) {
      std::cout << "The '" << Command << "' command is known as: '"
                << Names[0] << "'";
      for (unsigned i = 1, e = Names.size(); i != e; ++i)
        std::cout << ", '" << Names[i] << "'";
      std::cout << "\n";
    }

  } else {
    unsigned MaxSize = 0;
    for (std::map<std::string, CLICommand*>::iterator I = CommandTable.begin(),
           E = CommandTable.end(); I != E; ++I)
      if (I->first.size() > MaxSize &&
          I->first == I->second->getPrimaryOptionName())
        MaxSize = I->first.size();

    // Loop over all of the commands, printing the short help version
    for (std::map<std::string, CLICommand*>::iterator I = CommandTable.begin(),
           E = CommandTable.end(); I != E; ++I)
      if (I->first == I->second->getPrimaryOptionName())
        std::cout << I->first << std::string(MaxSize - I->first.size(), ' ')
                  << " - " << I->second->getShortHelp() << "\n";
  }
}
