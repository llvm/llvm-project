//===-- llvm-rtdyld.cpp - MCJIT Testing Tool ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a testing tool for use with the MC-JIT LLVM components.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/ObjectImage.h"
#include "llvm/ExecutionEngine/ObjectBuffer.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
using namespace llvm;
using namespace llvm::object;

static cl::list<std::string>
InputFileList(cl::Positional, cl::ZeroOrMore,
              cl::desc("<input file>"));

enum ActionType {
  AC_Execute
};

static cl::opt<ActionType>
Action(cl::desc("Action to perform:"),
       cl::init(AC_Execute),
       cl::values(clEnumValN(AC_Execute, "execute",
                             "Load, link, and execute the inputs."),
                  clEnumValEnd));

static cl::opt<std::string>
EntryPoint("entry",
           cl::desc("Function to call as entry point."),
           cl::init("_main"));

/* *** */

// A trivial memory manager that doesn't do anything fancy, just uses the
// support library allocation routines directly.
class TrivialMemoryManager : public RTDyldMemoryManager {
public:
  SmallVector<sys::MemoryBlock, 16> FunctionMemory;
  SmallVector<sys::MemoryBlock, 16> DataMemory;

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID);
  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID);

  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true) {
    return 0;
  }

  // Invalidate instruction cache for sections with execute permissions.
  // Some platforms with separate data cache and instruction cache require
  // explicit cache flush, otherwise JIT code manipulations (like resolved
  // relocations) will get to the data cache but not to the instruction cache.
  virtual void invalidateInstructionCache();
};

uint8_t *TrivialMemoryManager::allocateCodeSection(uintptr_t Size,
                                                   unsigned Alignment,
                                                   unsigned SectionID) {
  sys::MemoryBlock MB = sys::Memory::AllocateRWX(Size, 0, 0);
  FunctionMemory.push_back(MB);
  return (uint8_t*)MB.base();
}

uint8_t *TrivialMemoryManager::allocateDataSection(uintptr_t Size,
                                                   unsigned Alignment,
                                                   unsigned SectionID) {
  sys::MemoryBlock MB = sys::Memory::AllocateRWX(Size, 0, 0);
  DataMemory.push_back(MB);
  return (uint8_t*)MB.base();
}

void TrivialMemoryManager::invalidateInstructionCache() {
  for (int i = 0, e = FunctionMemory.size(); i != e; ++i)
    sys::Memory::InvalidateInstructionCache(FunctionMemory[i].base(),
                                            FunctionMemory[i].size());

  for (int i = 0, e = DataMemory.size(); i != e; ++i)
    sys::Memory::InvalidateInstructionCache(DataMemory[i].base(),
                                            DataMemory[i].size());
}

static const char *ProgramName;

static void Message(const char *Type, const Twine &Msg) {
  errs() << ProgramName << ": " << Type << ": " << Msg << "\n";
}

static int Error(const Twine &Msg) {
  Message("error", Msg);
  return 1;
}

/* *** */

static int executeInput() {
  // Instantiate a dynamic linker.
  TrivialMemoryManager *MemMgr = new TrivialMemoryManager;
  RuntimeDyld Dyld(MemMgr);

  // If we don't have any input files, read from stdin.
  if (!InputFileList.size())
    InputFileList.push_back("-");
  for(unsigned i = 0, e = InputFileList.size(); i != e; ++i) {
    // Load the input memory buffer.
    OwningPtr<MemoryBuffer> InputBuffer;
    OwningPtr<ObjectImage>  LoadedObject;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFileList[i],
                                                     InputBuffer))
      return Error("unable to read input: '" + ec.message() + "'");

    // Load the object file
    LoadedObject.reset(Dyld.loadObject(new ObjectBuffer(InputBuffer.take())));
    if (!LoadedObject) {
      return Error(Dyld.getErrorString());
    }
  }

  // Resolve all the relocations we can.
  Dyld.resolveRelocations();
  // Clear instruction cache before code will be executed.
  MemMgr->invalidateInstructionCache();

  // FIXME: Error out if there are unresolved relocations.

  // Get the address of the entry point (_main by default).
  void *MainAddress = Dyld.getSymbolAddress(EntryPoint);
  if (MainAddress == 0)
    return Error("no definition for '" + EntryPoint + "'");

  // Invalidate the instruction cache for each loaded function.
  for (unsigned i = 0, e = MemMgr->FunctionMemory.size(); i != e; ++i) {
    sys::MemoryBlock &Data = MemMgr->FunctionMemory[i];
    // Make sure the memory is executable.
    std::string ErrorStr;
    sys::Memory::InvalidateInstructionCache(Data.base(), Data.size());
    if (!sys::Memory::setExecutable(Data, &ErrorStr))
      return Error("unable to mark function executable: '" + ErrorStr + "'");
  }

  // Dispatch to _main().
  errs() << "loaded '" << EntryPoint << "' at: " << (void*)MainAddress << "\n";

  int (*Main)(int, const char**) =
    (int(*)(int,const char**)) uintptr_t(MainAddress);
  const char **Argv = new const char*[2];
  // Use the name of the first input object module as argv[0] for the target.
  Argv[0] = InputFileList[0].c_str();
  Argv[1] = 0;
  return Main(1, Argv);
}

int main(int argc, char **argv) {
  ProgramName = argv[0];
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "llvm MC-JIT tool\n");

  switch (Action) {
  case AC_Execute:
    return executeInput();
  }
}
