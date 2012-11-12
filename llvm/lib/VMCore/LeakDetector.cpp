//===-- LeakDetector.cpp - Implement LeakDetector interface ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LeakDetector class.
//
//===----------------------------------------------------------------------===//

#include "LLVMContextImpl.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Threading.h"
#include "llvm/Value.h"
using namespace llvm;

static ManagedStatic<sys::SmartMutex<true> > ObjectsLock;
static ManagedStatic<LeakDetectorImpl<void> > Objects;

static void clearGarbage(LLVMContext &Context) {
  Objects->clear();
  Context.pImpl->LLVMObjects.clear();
}

void LeakDetector::addGarbageObjectImpl(void *Object) {
  sys::SmartScopedLock<true> Lock(*ObjectsLock);
  Objects->addGarbage(Object);
}

void LeakDetector::addGarbageObjectImpl(const Value *Object) {
  LLVMContextImpl *pImpl = Object->getContext().pImpl;
  pImpl->LLVMObjects.addGarbage(Object);
}

void LeakDetector::removeGarbageObjectImpl(void *Object) {
  sys::SmartScopedLock<true> Lock(*ObjectsLock);
  Objects->removeGarbage(Object);
}

void LeakDetector::removeGarbageObjectImpl(const Value *Object) {
  LLVMContextImpl *pImpl = Object->getContext().pImpl;
  pImpl->LLVMObjects.removeGarbage(Object);
}

void LeakDetector::checkForGarbageImpl(LLVMContext &Context, 
                                       const std::string &Message) {
  LLVMContextImpl *pImpl = Context.pImpl;
  sys::SmartScopedLock<true> Lock(*ObjectsLock);
  
  Objects->setName("GENERIC");
  pImpl->LLVMObjects.setName("LLVM");
  
  // use non-short-circuit version so that both checks are performed
  if (Objects->hasGarbage(Message) |
      pImpl->LLVMObjects.hasGarbage(Message))
    errs() << "\nThis is probably because you removed an object, but didn't "
           << "delete it.  Please check your code for memory leaks.\n";

  // Clear out results so we don't get duplicate warnings on
  // next call...
  clearGarbage(Context);
}
