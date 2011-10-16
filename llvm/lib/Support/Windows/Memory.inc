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

#include "Windows.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Process.h"

namespace llvm {
using namespace sys;

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

MemoryBlock Memory::AllocateRWX(size_t NumBytes,
                                const MemoryBlock *NearBlock,
                                std::string *ErrMsg) {
  if (NumBytes == 0) return MemoryBlock();

  static const size_t pageSize = Process::GetPageSize();
  size_t NumPages = (NumBytes+pageSize-1)/pageSize;

  PVOID start = NearBlock ? static_cast<unsigned char *>(NearBlock->base()) +
                                NearBlock->size() : NULL;

  void *pa = VirtualAlloc(start, NumPages*pageSize, MEM_RESERVE | MEM_COMMIT,
                  PAGE_EXECUTE_READWRITE);
  if (pa == NULL) {
    if (NearBlock) {
      // Try again without the NearBlock hint
      return AllocateRWX(NumBytes, NULL, ErrMsg);
    }
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

static DWORD getProtection(const void *addr) {
  MEMORY_BASIC_INFORMATION info;
  if (sizeof(info) == ::VirtualQuery(addr, &info, sizeof(info))) {
    return info.Protect;
  }
  return 0;
}

bool Memory::setWritable(MemoryBlock &M, std::string *ErrMsg) {
  if (!setRangeWritable(M.Address, M.Size)) {
    return MakeErrMsg(ErrMsg, "Cannot set memory to writeable: ");
  }
  return true;
}

bool Memory::setExecutable(MemoryBlock &M, std::string *ErrMsg) {
  if (!setRangeExecutable(M.Address, M.Size)) {
    return MakeErrMsg(ErrMsg, "Cannot set memory to executable: ");
  }
  return true;
}

bool Memory::setRangeWritable(const void *Addr, size_t Size) {
  DWORD prot = getProtection(Addr);
  if (!prot)
    return false;

  if (prot == PAGE_EXECUTE || prot == PAGE_EXECUTE_READ) {
    prot = PAGE_EXECUTE_READWRITE;
  } else if (prot == PAGE_NOACCESS || prot == PAGE_READONLY) {
    prot = PAGE_READWRITE;
  }

  DWORD oldProt;
  sys::Memory::InvalidateInstructionCache(Addr, Size);
  return ::VirtualProtect(const_cast<LPVOID>(Addr), Size, prot, &oldProt)
            == TRUE;
}

bool Memory::setRangeExecutable(const void *Addr, size_t Size) {
  DWORD prot = getProtection(Addr);
  if (!prot)
    return false;

  if (prot == PAGE_NOACCESS) {
    prot = PAGE_EXECUTE;
  } else if (prot == PAGE_READONLY) {
    prot = PAGE_EXECUTE_READ;
  } else if (prot == PAGE_READWRITE) {
    prot = PAGE_EXECUTE_READWRITE;
  }

  DWORD oldProt;
  sys::Memory::InvalidateInstructionCache(Addr, Size);
  return ::VirtualProtect(const_cast<LPVOID>(Addr), Size, prot, &oldProt)
            == TRUE;
}

}
