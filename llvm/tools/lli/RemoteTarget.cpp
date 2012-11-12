//===- RemoteTarget.cpp - LLVM Remote process JIT execution --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the RemoteTarget class which executes JITed code in a
// separate address range from where it was built.
//
//===----------------------------------------------------------------------===//

#include "RemoteTarget.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Memory.h"
#include <stdlib.h>
#include <string>
using namespace llvm;

bool RemoteTarget::allocateSpace(size_t Size, unsigned Alignment,
                                 uint64_t &Address) {
  sys::MemoryBlock *Prev = Allocations.size() ? &Allocations.back() : NULL;
  sys::MemoryBlock Mem = sys::Memory::AllocateRWX(Size, Prev, &ErrorMsg);
  if (Mem.base() == NULL)
    return true;
  if ((uintptr_t)Mem.base() % Alignment) {
    ErrorMsg = "unable to allocate sufficiently aligned memory";
    return true;
  }
  Address = reinterpret_cast<uint64_t>(Mem.base());
  return false;
}

bool RemoteTarget::loadData(uint64_t Address, const void *Data, size_t Size) {
  memcpy ((void*)Address, Data, Size);
  return false;
}

bool RemoteTarget::loadCode(uint64_t Address, const void *Data, size_t Size) {
  memcpy ((void*)Address, Data, Size);
  sys::MemoryBlock Mem((void*)Address, Size);
  sys::Memory::setExecutable(Mem, &ErrorMsg);
  return false;
}

bool RemoteTarget::executeCode(uint64_t Address, int &RetVal) {
  int (*fn)(void) = (int(*)(void))Address;
  RetVal = fn();
  return false;
}

void RemoteTarget::create() {
}

void RemoteTarget::stop() {
  for (unsigned i = 0, e = Allocations.size(); i != e; ++i)
    sys::Memory::ReleaseRWX(Allocations[i]);
}
