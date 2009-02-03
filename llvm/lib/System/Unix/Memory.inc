//===- Unix/Memory.cpp - Generic UNIX System Configuration ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some functions for various memory management utilities.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/System/Process.h"

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#endif

/// AllocateRWX - Allocate a slab of memory with read/write/execute
/// permissions.  This is typically used for JIT applications where we want
/// to emit code to the memory then jump to it.  Getting this type of memory
/// is very OS specific.
///
llvm::sys::MemoryBlock 
llvm::sys::Memory::AllocateRWX(unsigned NumBytes, const MemoryBlock* NearBlock,
                               std::string *ErrMsg) {
  if (NumBytes == 0) return MemoryBlock();

  unsigned pageSize = Process::GetPageSize();
  unsigned NumPages = (NumBytes+pageSize-1)/pageSize;

  int fd = -1;
#ifdef NEED_DEV_ZERO_FOR_MMAP
  static int zero_fd = open("/dev/zero", O_RDWR);
  if (zero_fd == -1) {
    MakeErrMsg(ErrMsg, "Can't open /dev/zero device");
    return MemoryBlock();
  }
  fd = zero_fd;
#endif

  int flags = MAP_PRIVATE |
#ifdef HAVE_MMAP_ANONYMOUS
  MAP_ANONYMOUS
#else
  MAP_ANON
#endif
  ;

  void* start = NearBlock ? (unsigned char*)NearBlock->base() + 
                            NearBlock->size() : 0;

#if defined(__APPLE__) && defined(__arm__)
  void *pa = ::mmap(start, pageSize*NumPages, PROT_READ|PROT_EXEC,
                    flags, fd, 0);
#else
  void *pa = ::mmap(start, pageSize*NumPages, PROT_READ|PROT_WRITE|PROT_EXEC,
                    flags, fd, 0);
#endif
  if (pa == MAP_FAILED) {
    if (NearBlock) //Try again without a near hint
      return AllocateRWX(NumBytes, 0);

    MakeErrMsg(ErrMsg, "Can't allocate RWX Memory");
    return MemoryBlock();
  }

#if defined(__APPLE__) && defined(__arm__)
  kern_return_t kr = vm_protect(mach_task_self(), (vm_address_t)pa,
                                (vm_size_t)(pageSize*NumPages), 0,
                                VM_PROT_READ | VM_PROT_EXECUTE | VM_PROT_COPY);
  if (KERN_SUCCESS != kr) {
    MakeErrMsg(ErrMsg, "vm_protect max RX failed\n");
    return sys::MemoryBlock();
  }

  kr = vm_protect(mach_task_self(), (vm_address_t)pa,
                  (vm_size_t)(pageSize*NumPages), 0,
                  VM_PROT_READ | VM_PROT_WRITE);
  if (KERN_SUCCESS != kr) {
    MakeErrMsg(ErrMsg, "vm_protect RW failed\n");
    return sys::MemoryBlock();
  }
#endif

  MemoryBlock result;
  result.Address = pa;
  result.Size = NumPages*pageSize;

  return result;
}

bool llvm::sys::Memory::ReleaseRWX(MemoryBlock &M, std::string *ErrMsg) {
  if (M.Address == 0 || M.Size == 0) return false;
  if (0 != ::munmap(M.Address, M.Size))
    return MakeErrMsg(ErrMsg, "Can't release RWX Memory");
  return false;
}

bool llvm::sys::Memory::setWritable (MemoryBlock &M, std::string *ErrMsg) {
#if defined(__APPLE__) && defined(__arm__)
  if (M.Address == 0 || M.Size == 0) return false;
  sys::Memory::InvalidateInstructionCache(M.Address, M.Size);
  kern_return_t kr = vm_protect(mach_task_self(), (vm_address_t)M.Address,
    (vm_size_t)M.Size, 0, VM_PROT_READ | VM_PROT_WRITE);
  return KERN_SUCCESS == kr;
#else
  return true;
#endif
}

bool llvm::sys::Memory::setExecutable (MemoryBlock &M, std::string *ErrMsg) {
#if defined(__APPLE__) && defined(__arm__)
  if (M.Address == 0 || M.Size == 0) return false;
  sys::Memory::InvalidateInstructionCache(M.Address, M.Size);
  kern_return_t kr = vm_protect(mach_task_self(), (vm_address_t)M.Address,
    (vm_size_t)M.Size, 0, VM_PROT_READ | VM_PROT_EXECUTE | VM_PROT_COPY);
  return KERN_SUCCESS == kr;
#else
  return false;
#endif
}

bool llvm::sys::Memory::setRangeWritable(const void *Addr, size_t Size) {
#if defined(__APPLE__) && defined(__arm__)
  kern_return_t kr = vm_protect(mach_task_self(), (vm_address_t)Addr,
                                (vm_size_t)Size, 0,
                                VM_PROT_READ | VM_PROT_WRITE);
  return KERN_SUCCESS == kr;
#else
  return true;
#endif
}

bool llvm::sys::Memory::setRangeExecutable(const void *Addr, size_t Size) {
#if defined(__APPLE__) && defined(__arm__)
  kern_return_t kr = vm_protect(mach_task_self(), (vm_address_t)Addr,
                                (vm_size_t)Size, 0,
                                VM_PROT_READ | VM_PROT_EXECUTE | VM_PROT_COPY);
  return KERN_SUCCESS == kr;
#else
  return true;
#endif
}
