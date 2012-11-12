//===-- OProfileJITEventListener.cpp - Tell OProfile about JITted code ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a JITEventListener object that uses OProfileWrapper to tell
// oprofile about JITted functions, including source line information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/ExecutionEngine/JITEventListener.h"

#define DEBUG_TYPE "oprofile-jit-event-listener"
#include "llvm/DebugInfo.h"
#include "llvm/Function.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/ExecutionEngine/OProfileWrapper.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Errno.h"
#include "EventListenerCommon.h"

#include <dirent.h>
#include <fcntl.h>

using namespace llvm;
using namespace llvm::jitprofiling;

namespace {

class OProfileJITEventListener : public JITEventListener {
  OProfileWrapper& Wrapper;

  void initialize();

public:
  OProfileJITEventListener(OProfileWrapper& LibraryWrapper)
  : Wrapper(LibraryWrapper) {
    initialize();
  }

  ~OProfileJITEventListener();

  virtual void NotifyFunctionEmitted(const Function &F,
                                void *FnStart, size_t FnSize,
                                const JITEvent_EmittedFunctionDetails &Details);

  virtual void NotifyFreeingMachineCode(void *OldPtr);
};

void OProfileJITEventListener::initialize() {
  if (!Wrapper.op_open_agent()) {
    const std::string err_str = sys::StrError();
    DEBUG(dbgs() << "Failed to connect to OProfile agent: " << err_str << "\n");
  } else {
    DEBUG(dbgs() << "Connected to OProfile agent.\n");
  }
}

OProfileJITEventListener::~OProfileJITEventListener() {
  if (Wrapper.isAgentAvailable()) {
    if (Wrapper.op_close_agent() == -1) {
      const std::string err_str = sys::StrError();
      DEBUG(dbgs() << "Failed to disconnect from OProfile agent: "
                   << err_str << "\n");
    } else {
      DEBUG(dbgs() << "Disconnected from OProfile agent.\n");
    }
  }
}

static debug_line_info LineStartToOProfileFormat(
    const MachineFunction &MF, FilenameCache &Filenames,
    uintptr_t Address, DebugLoc Loc) {
  debug_line_info Result;
  Result.vma = Address;
  Result.lineno = Loc.getLine();
  Result.filename = Filenames.getFilename(
    Loc.getScope(MF.getFunction()->getContext()));
  DEBUG(dbgs() << "Mapping " << reinterpret_cast<void*>(Result.vma) << " to "
               << Result.filename << ":" << Result.lineno << "\n");
  return Result;
}

// Adds the just-emitted function to the symbol table.
void OProfileJITEventListener::NotifyFunctionEmitted(
    const Function &F, void *FnStart, size_t FnSize,
    const JITEvent_EmittedFunctionDetails &Details) {
  assert(F.hasName() && FnStart != 0 && "Bad symbol to add");
  if (Wrapper.op_write_native_code(F.getName().data(),
                           reinterpret_cast<uint64_t>(FnStart),
                           FnStart, FnSize) == -1) {
    DEBUG(dbgs() << "Failed to tell OProfile about native function "
          << F.getName() << " at ["
          << FnStart << "-" << ((char*)FnStart + FnSize) << "]\n");
    return;
  }

  if (!Details.LineStarts.empty()) {
    // Now we convert the line number information from the address/DebugLoc
    // format in Details to the address/filename/lineno format that OProfile
    // expects.  Note that OProfile 0.9.4 has a bug that causes it to ignore
    // line numbers for addresses above 4G.
    FilenameCache Filenames;
    std::vector<debug_line_info> LineInfo;
    LineInfo.reserve(1 + Details.LineStarts.size());

    DebugLoc FirstLoc = Details.LineStarts[0].Loc;
    assert(!FirstLoc.isUnknown()
           && "LineStarts should not contain unknown DebugLocs");
    MDNode *FirstLocScope = FirstLoc.getScope(F.getContext());
    DISubprogram FunctionDI = getDISubprogram(FirstLocScope);
    if (FunctionDI.Verify()) {
      // If we have debug info for the function itself, use that as the line
      // number of the first several instructions.  Otherwise, after filling
      // LineInfo, we'll adjust the address of the first line number to point at
      // the start of the function.
      debug_line_info line_info;
      line_info.vma = reinterpret_cast<uintptr_t>(FnStart);
      line_info.lineno = FunctionDI.getLineNumber();
      line_info.filename = Filenames.getFilename(FirstLocScope);
      LineInfo.push_back(line_info);
    }

    for (std::vector<EmittedFunctionDetails::LineStart>::const_iterator
           I = Details.LineStarts.begin(), E = Details.LineStarts.end();
         I != E; ++I) {
      LineInfo.push_back(LineStartToOProfileFormat(
                           *Details.MF, Filenames, I->Address, I->Loc));
    }

    // In case the function didn't have line info of its own, adjust the first
    // line info's address to include the start of the function.
    LineInfo[0].vma = reinterpret_cast<uintptr_t>(FnStart);

    if (Wrapper.op_write_debug_line_info(FnStart, LineInfo.size(),
                                      &*LineInfo.begin()) == -1) {
      DEBUG(dbgs()
            << "Failed to tell OProfile about line numbers for native function "
            << F.getName() << " at ["
            << FnStart << "-" << ((char*)FnStart + FnSize) << "]\n");
    }
  }
}

// Removes the being-deleted function from the symbol table.
void OProfileJITEventListener::NotifyFreeingMachineCode(void *FnStart) {
  assert(FnStart && "Invalid function pointer");
  if (Wrapper.op_unload_native_code(reinterpret_cast<uint64_t>(FnStart)) == -1) {
    DEBUG(dbgs()
          << "Failed to tell OProfile about unload of native function at "
          << FnStart << "\n");
  }
}

}  // anonymous namespace.

namespace llvm {
JITEventListener *JITEventListener::createOProfileJITEventListener() {
  static OwningPtr<OProfileWrapper> JITProfilingWrapper(new OProfileWrapper);
  return new OProfileJITEventListener(*JITProfilingWrapper);
}

// for testing
JITEventListener *JITEventListener::createOProfileJITEventListener(
                                      OProfileWrapper* TestImpl) {
  return new OProfileJITEventListener(*TestImpl);
}

} // namespace llvm

