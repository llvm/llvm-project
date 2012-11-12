//===-- ManagedStatic.cpp - Static Global wrapper -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ManagedStatic class and llvm_shutdown().
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Atomic.h"
#include <cassert>
using namespace llvm;

static const ManagedStaticBase *StaticList = 0;

void ManagedStaticBase::RegisterManagedStatic(void *(*Creator)(),
                                              void (*Deleter)(void*)) const {
  if (llvm_is_multithreaded()) {
    llvm_acquire_global_lock();

    if (Ptr == 0) {
      void* tmp = Creator ? Creator() : 0;

      TsanHappensBefore(this);
      sys::MemoryFence();

      // This write is racy against the first read in the ManagedStatic
      // accessors. The race is benign because it does a second read after a
      // memory fence, at which point it isn't possible to get a partial value.
      TsanIgnoreWritesBegin();
      Ptr = tmp;
      TsanIgnoreWritesEnd();
      DeleterFn = Deleter;
      
      // Add to list of managed statics.
      Next = StaticList;
      StaticList = this;
    }

    llvm_release_global_lock();
  } else {
    assert(Ptr == 0 && DeleterFn == 0 && Next == 0 &&
           "Partially initialized ManagedStatic!?");
    Ptr = Creator ? Creator() : 0;
    DeleterFn = Deleter;
  
    // Add to list of managed statics.
    Next = StaticList;
    StaticList = this;
  }
}

void ManagedStaticBase::destroy() const {
  assert(DeleterFn && "ManagedStatic not initialized correctly!");
  assert(StaticList == this &&
         "Not destroyed in reverse order of construction?");
  // Unlink from list.
  StaticList = Next;
  Next = 0;

  // Destroy memory.
  DeleterFn(Ptr);
  
  // Cleanup.
  Ptr = 0;
  DeleterFn = 0;
}

/// llvm_shutdown - Deallocate and destroy all ManagedStatic variables.
void llvm::llvm_shutdown() {
  while (StaticList)
    StaticList->destroy();

  if (llvm_is_multithreaded()) llvm_stop_multithreaded();
}
