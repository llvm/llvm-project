//===-- SectionMemoryManager.h - Memory allocator for MCJIT -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of a section-based memory manager used by
// the MCJIT execution engine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTION_ENGINE_SECTION_MEMORY_MANAGER_H
#define LLVM_EXECUTION_ENGINE_SECTION_MEMORY_MANAGER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/JITMemoryManager.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Memory.h"

namespace llvm {

// Section-based memory manager for MCJIT
class SectionMemoryManager : public JITMemoryManager {

public:

  SectionMemoryManager() { }
  ~SectionMemoryManager();

  virtual uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID);

  virtual uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                       unsigned SectionID);

  virtual void *getPointerToNamedFunction(const std::string &Name,
                                          bool AbortOnFailure = true);

  // Invalidate instruction cache for code sections. Some platforms with
  // separate data cache and instruction cache require explicit cache flush,
  // otherwise JIT code manipulations (like resolved relocations) will get to
  // the data cache but not to the instruction cache.
  virtual void invalidateInstructionCache();

private:

  SmallVector<sys::MemoryBlock, 16> AllocatedDataMem;
  SmallVector<sys::MemoryBlock, 16> AllocatedCodeMem;
  SmallVector<sys::MemoryBlock, 16> FreeCodeMem;

public:

  ///
  /// Functions below are not used by MCJIT, but must be implemented because
  /// they are declared as pure virtuals in the base class.
  ///

  virtual void setMemoryWritable() {
    llvm_unreachable("Unexpected call!");
  }
  virtual void setMemoryExecutable() {
    llvm_unreachable("Unexpected call!");
  }
  virtual void setPoisonMemory(bool poison) {
    llvm_unreachable("Unexpected call!");
  }
  virtual void AllocateGOT() {
    llvm_unreachable("Unexpected call!");
  }
  virtual uint8_t *getGOTBase() const {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual uint8_t *startFunctionBody(const Function *F,
                                     uintptr_t &ActualSize){
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual uint8_t *allocateStub(const GlobalValue* F, unsigned StubSize,
                                unsigned Alignment) {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual void endFunctionBody(const Function *F, uint8_t *FunctionStart,
                               uint8_t *FunctionEnd) {
    llvm_unreachable("Unexpected call!");
  }
  virtual uint8_t *allocateSpace(intptr_t Size, unsigned Alignment) {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment) {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual void deallocateFunctionBody(void *Body) {
    llvm_unreachable("Unexpected call!");
  }
  virtual uint8_t *startExceptionTable(const Function *F,
                                       uintptr_t &ActualSize) {
    llvm_unreachable("Unexpected call!");
    return 0;
  }
  virtual void endExceptionTable(const Function *F, uint8_t *TableStart,
                                 uint8_t *TableEnd, uint8_t *FrameRegister) {
    llvm_unreachable("Unexpected call!");
  }
  virtual void deallocateExceptionTable(void *ET) {
    llvm_unreachable("Unexpected call!");
  }
};

}

#endif // LLVM_EXECUTION_ENGINE_SECTION_MEMORY_MANAGER_H
