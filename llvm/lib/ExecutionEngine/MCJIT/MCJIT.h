//===-- MCJIT.h - Class definition for the MCJIT ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_MCJIT_H
#define LLVM_LIB_EXECUTIONENGINE_MCJIT_H

#include "llvm/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"

namespace llvm {

class ObjectImage;

// FIXME: This makes all kinds of horrible assumptions for the time being,
// like only having one module, not needing to worry about multi-threading,
// blah blah. Purely in get-it-up-and-limping mode for now.

class MCJIT : public ExecutionEngine {
  MCJIT(Module *M, TargetMachine *tm, RTDyldMemoryManager *MemMgr,
        bool AllocateGVsWithCode);

  TargetMachine *TM;
  MCContext *Ctx;
  RTDyldMemoryManager *MemMgr;
  RuntimeDyld Dyld;
  SmallVector<JITEventListener*, 2> EventListeners;

  // FIXME: Add support for multiple modules
  bool isCompiled;
  Module *M;
  OwningPtr<ObjectImage> LoadedObject;

public:
  ~MCJIT();

  /// @name ExecutionEngine interface implementation
  /// @{

  virtual void finalizeObject();

  virtual void *getPointerToBasicBlock(BasicBlock *BB);

  virtual void *getPointerToFunction(Function *F);

  virtual void *recompileAndRelinkFunction(Function *F);

  virtual void freeMachineCodeForFunction(Function *F);

  virtual GenericValue runFunction(Function *F,
                                   const std::vector<GenericValue> &ArgValues);

  /// getPointerToNamedFunction - This method returns the address of the
  /// specified function by using the dlsym function call.  As such it is only
  /// useful for resolving library symbols, not code generated symbols.
  ///
  /// If AbortOnFailure is false and no function with the given name is
  /// found, this function silently returns a null pointer. Otherwise,
  /// it prints a message to stderr and aborts.
  ///
  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);

  /// mapSectionAddress - map a section to its target address space value.
  /// Map the address of a JIT section as returned from the memory manager
  /// to the address in the target process as the running code will see it.
  /// This is the address which will be used for relocation resolution.
  virtual void mapSectionAddress(const void *LocalAddress,
                                 uint64_t TargetAddress) {
    Dyld.mapSectionAddress(LocalAddress, TargetAddress);
  }

  virtual void RegisterJITEventListener(JITEventListener *L);
  virtual void UnregisterJITEventListener(JITEventListener *L);

  /// @}
  /// @name (Private) Registration Interfaces
  /// @{

  static void Register() {
    MCJITCtor = createJIT;
  }

  static ExecutionEngine *createJIT(Module *M,
                                    std::string *ErrorStr,
                                    JITMemoryManager *JMM,
                                    bool GVsWithCode,
                                    TargetMachine *TM);

  // @}

protected:
  /// emitObject -- Generate a JITed object in memory from the specified module
  /// Currently, MCJIT only supports a single module and the module passed to
  /// this function call is expected to be the contained module.  The module
  /// is passed as a parameter here to prepare for multiple module support in 
  /// the future.
  void emitObject(Module *M);

  void NotifyObjectEmitted(const ObjectImage& Obj);
  void NotifyFreeingObject(const ObjectImage& Obj);
};

} // End llvm namespace

#endif
