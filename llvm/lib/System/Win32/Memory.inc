//===- Win32/Memory.cpp - Win32 Memory Implementation -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of various Memory
// management utilities
//
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include "llvm/System/Process.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

MemoryBlock Memory::AllocateRWX(unsigned NumBytes,
                                const MemoryBlock *NearBlock,
                                std::string *ErrMsg) {
  if (NumBytes == 0) return MemoryBlock();

  static const long pageSize = Process::GetPageSize();
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;

  //FIXME: support NearBlock if ever needed on Win64.

  void *pa = VirtualAlloc(NULL, NumPages*pageSize, MEM_COMMIT,
                  PAGE_EXECUTE_READWRITE);
  if (pa == NULL) {
    MakeErrMsg(ErrMsg, "Can't allocate RWX Memory: ");
    return MemoryBlock();
  }

  MemoryBlock result;
  result.Address = pa;
  result.Size = NumPages*pageSize;
  return result;
}

bool Memory::ReleaseRWX(MemoryBlock &M, std::string *ErrMsg) {
  if (M.Address == 0 || M.Size == 0) return false;
  if (!VirtualFree(M.Address, 0, MEM_RELEASE))
    return MakeErrMsg(ErrMsg, "Can't release RWX Memory: ");
  return false;
}

bool Memory::setWritable(MemoryBlock &M, std::string *ErrMsg) {
  return true;
}

bool Memory::setExecutable(MemoryBlock &M, std::string *ErrMsg) {
  return false;
}

bool Memory::setRangeWritable(const void *Addr, size_t Size) {
  return true;
}

bool Memory::setRangeExecutable(const void *Addr, size_t Size) {
  return false;
}

}
