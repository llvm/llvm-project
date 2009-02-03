//===-- Debugger.cpp - LLVM debugger library implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the main implementation of the LLVM debugger library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/Debugger.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Debugger/InferiorProcess.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringExtras.h"
#include <cstdlib>
#include <memory>
using namespace llvm;

/// Debugger constructor - Initialize the debugger to its initial, empty, state.
///
Debugger::Debugger() : Environment(0), Program(0), Process(0) {
}

Debugger::~Debugger() {
  // Killing the program could throw an exception.  We don't want to progagate
  // the exception out of our destructor though.
  try {
    killProgram();
  } catch (const char *) {
  } catch (const std::string &) {
  }

  unloadProgram();
}

/// getProgramPath - Get the path of the currently loaded program, or an
/// empty string if none is loaded.
std::string Debugger::getProgramPath() const {
  return Program ? Program->getModuleIdentifier() : "";
}

static Module *
getMaterializedModuleProvider(const std::string &Filename) {
  std::auto_ptr<MemoryBuffer> Buffer;
  Buffer.reset(MemoryBuffer::getFileOrSTDIN(Filename.c_str()));
  if (Buffer.get())
    return ParseBitcodeFile(Buffer.get());
  return 0;
}

/// loadProgram - If a program is currently loaded, unload it.  Then search
/// the PATH for the specified program, loading it when found.  If the
/// specified program cannot be found, an exception is thrown to indicate the
/// error.
void Debugger::loadProgram(const std::string &Filename) {
  if ((Program = getMaterializedModuleProvider(Filename)) ||
      (Program = getMaterializedModuleProvider(Filename+".bc")))
    return;   // Successfully loaded the program.

  // Search the program path for the file...
  if (const char *PathS = getenv("PATH")) {
    std::string Path = PathS;

    std::string Directory = getToken(Path, ":");
    while (!Directory.empty()) {
      if ((Program = getMaterializedModuleProvider(Directory +"/"+ Filename)) ||
          (Program = getMaterializedModuleProvider(Directory +"/"+ Filename
                                                                      + ".bc")))
        return;   // Successfully loaded the program.

      Directory = getToken(Path, ":");
    }
  }

  throw "Could not find program '" + Filename + "'!";
}

/// unloadProgram - If a program is running, kill it, then unload all traces
/// of the current program.  If no program is loaded, this method silently
/// succeeds.
void Debugger::unloadProgram() {
  if (!isProgramLoaded()) return;
  killProgram();
  delete Program;
  Program = 0;
}


/// createProgram - Create an instance of the currently loaded program,
/// killing off any existing one.  This creates the program and stops it at
/// the first possible moment.  If there is no program loaded or if there is a
/// problem starting the program, this method throws an exception.
void Debugger::createProgram() {
  if (!isProgramLoaded())
    throw "Cannot start program: none is loaded.";

  // Kill any existing program.
  killProgram();

  // Add argv[0] to the arguments vector..
  std::vector<std::string> Args(ProgramArguments);
  Args.insert(Args.begin(), getProgramPath());

  // Start the new program... this could throw if the program cannot be started.
  Process = InferiorProcess::create(Program, Args, Environment);
}

InferiorProcess *
InferiorProcess::create(Module *M, const std::vector<std::string> &Arguments,
                        const char * const *envp) {
  throw"No supported binding to inferior processes (debugger not implemented).";
}

/// killProgram - If the program is currently executing, kill off the
/// process and free up any state related to the currently running program.  If
/// there is no program currently running, this just silently succeeds.
void Debugger::killProgram() {
  // The destructor takes care of the dirty work.
  try {
    delete Process;
  } catch (...) {
    Process = 0;
    throw;
  }
  Process = 0;
}

/// stepProgram - Implement the 'step' command, continuing execution until
/// the next possible stop point.
void Debugger::stepProgram() {
  assert(isProgramRunning() && "Cannot step if the program isn't running!");
  try {
    Process->stepProgram();
  } catch (InferiorProcessDead &IPD) {
    killProgram();
    throw NonErrorException("The program stopped with exit code " +
                            itostr(IPD.getExitCode()));
  } catch (...) {
    killProgram();
    throw;
  }
}

/// nextProgram - Implement the 'next' command, continuing execution until
/// the next possible stop point that is in the current function.
void Debugger::nextProgram() {
  assert(isProgramRunning() && "Cannot next if the program isn't running!");
  try {
    // This should step the process.  If the process enters a function, then it
    // should 'finish' it.  However, figuring this out is tricky.  In
    // particular, the program can do any of:
    //  0. Not change current frame.
    //  1. Entering or exiting a region within the current function
    //     (which changes the frame ID, but which we shouldn't 'finish')
    //  2. Exiting the current function (which changes the frame ID)
    //  3. Entering a function (which should be 'finish'ed)
    // For this reason, we have to be very careful about when we decide to do
    // the 'finish'.

    // Get the current frame, but don't trust it.  It could change...
    void *CurrentFrame = Process->getPreviousFrame(0);

    // Don't trust the current frame: get the caller frame.
    void *ParentFrame  = Process->getPreviousFrame(CurrentFrame);

    // Ok, we have some information, run the program one step.
    Process->stepProgram();

    // Where is the new frame?  The most common case, by far is that it has not
    // been modified (Case #0), in which case we don't need to do anything more.
    void *NewFrame = Process->getPreviousFrame(0);
    if (NewFrame != CurrentFrame) {
      // Ok, the frame changed.  If we are case #1, then the parent frame will
      // be identical.
      void *NewParentFrame = Process->getPreviousFrame(NewFrame);
      if (ParentFrame != NewParentFrame) {
        // Ok, now we know we aren't case #0 or #1.  Check to see if we entered
        // a new function.  If so, the parent frame will be "CurrentFrame".
        if (CurrentFrame == NewParentFrame)
          Process->finishProgram(NewFrame);
      }
    }

  } catch (InferiorProcessDead &IPD) {
    killProgram();
    throw NonErrorException("The program stopped with exit code " +
                            itostr(IPD.getExitCode()));
  } catch (...) {
    killProgram();
    throw;
  }
}

/// finishProgram - Implement the 'finish' command, continuing execution
/// until the specified frame ID returns.
void Debugger::finishProgram(void *Frame) {
  assert(isProgramRunning() && "Cannot cont if the program isn't running!");
  try {
    Process->finishProgram(Frame);
  } catch (InferiorProcessDead &IPD) {
    killProgram();
    throw NonErrorException("The program stopped with exit code " +
                            itostr(IPD.getExitCode()));
  } catch (...) {
    killProgram();
    throw;
  }
}

/// contProgram - Implement the 'cont' command, continuing execution until
/// the next breakpoint is encountered.
void Debugger::contProgram() {
  assert(isProgramRunning() && "Cannot cont if the program isn't running!");
  try {
    Process->contProgram();
  } catch (InferiorProcessDead &IPD) {
    killProgram();
    throw NonErrorException("The program stopped with exit code " +
                            itostr(IPD.getExitCode()));
  } catch (...) {
    killProgram();
    throw;
  }
}
