//===-- SectionMemoryManager.cpp - The memory manager for MCJIT -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the implementation of the section-based memory manager
// used by MCJIT.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MathExtras.h"

#include "SectionMemoryManager.h"

#ifdef __linux__
// These includes used by SectionMemoryManager::getPointerToNamedFunction()
// for Glibc trickery. Look comments in this function for more information.
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#include <fcntl.h>
#include <unistd.h>
#endif

namespace llvm {

uint8_t *SectionMemoryManager::allocateDataSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID) {
  if (!Alignment)
    Alignment = 16;
  // Ensure that enough memory is requested to allow aligning.
  size_t NumElementsAligned = 1 + (Size + Alignment - 1)/Alignment;
  uint8_t *Addr = (uint8_t*)calloc(NumElementsAligned, Alignment);

  // Honour the alignment requirement.
  uint8_t *AlignedAddr = (uint8_t*)RoundUpToAlignment((uint64_t)Addr, Alignment);

  // Store the original address from calloc so we can free it later.
  AllocatedDataMem.push_back(sys::MemoryBlock(Addr, NumElementsAligned*Alignment));
  return AlignedAddr;
}

uint8_t *SectionMemoryManager::allocateCodeSection(uintptr_t Size,
                                                    unsigned Alignment,
                                                    unsigned SectionID) {
  if (!Alignment)
    Alignment = 16;
  unsigned NeedAllocate = Alignment * ((Size + Alignment - 1)/Alignment + 1);
  uintptr_t Addr = 0;
  // Look in the list of free code memory regions and use a block there if one
  // is available.
  for (int i = 0, e = FreeCodeMem.size(); i != e; ++i) {
    sys::MemoryBlock &MB = FreeCodeMem[i];
    if (MB.size() >= NeedAllocate) {
      Addr = (uintptr_t)MB.base();
      uintptr_t EndOfBlock = Addr + MB.size();
      // Align the address.
      Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);
      // Store cutted free memory block.
      FreeCodeMem[i] = sys::MemoryBlock((void*)(Addr + Size),
                                        EndOfBlock - Addr - Size);
      return (uint8_t*)Addr;
    }
  }

  // No pre-allocated free block was large enough. Allocate a new memory region.
  sys::MemoryBlock MB = sys::Memory::AllocateRWX(NeedAllocate, 0, 0);

  AllocatedCodeMem.push_back(MB);
  Addr = (uintptr_t)MB.base();
  uintptr_t EndOfBlock = Addr + MB.size();
  // Align the address.
  Addr = (Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1);
  // The AllocateRWX may allocate much more memory than we need. In this case,
  // we store the unused memory as a free memory block.
  unsigned FreeSize = EndOfBlock-Addr-Size;
  if (FreeSize > 16)
    FreeCodeMem.push_back(sys::MemoryBlock((void*)(Addr + Size), FreeSize));

  // Return aligned address
  return (uint8_t*)Addr;
}

void SectionMemoryManager::invalidateInstructionCache() {
  for (int i = 0, e = AllocatedCodeMem.size(); i != e; ++i)
    sys::Memory::InvalidateInstructionCache(AllocatedCodeMem[i].base(),
                                            AllocatedCodeMem[i].size());
}

void *SectionMemoryManager::getPointerToNamedFunction(const std::string &Name,
                                                       bool AbortOnFailure) {
#if defined(__linux__)
  //===--------------------------------------------------------------------===//
  // Function stubs that are invoked instead of certain library calls
  //
  // Force the following functions to be linked in to anything that uses the
  // JIT. This is a hack designed to work around the all-too-clever Glibc
  // strategy of making these functions work differently when inlined vs. when
  // not inlined, and hiding their real definitions in a separate archive file
  // that the dynamic linker can't see. For more info, search for
  // 'libc_nonshared.a' on Google, or read http://llvm.org/PR274.
  if (Name == "stat") return (void*)(intptr_t)&stat;
  if (Name == "fstat") return (void*)(intptr_t)&fstat;
  if (Name == "lstat") return (void*)(intptr_t)&lstat;
  if (Name == "stat64") return (void*)(intptr_t)&stat64;
  if (Name == "fstat64") return (void*)(intptr_t)&fstat64;
  if (Name == "lstat64") return (void*)(intptr_t)&lstat64;
  if (Name == "atexit") return (void*)(intptr_t)&atexit;
  if (Name == "mknod") return (void*)(intptr_t)&mknod;
#endif // __linux__

  const char *NameStr = Name.c_str();
  void *Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr);
  if (Ptr) return Ptr;

  // If it wasn't found and if it starts with an underscore ('_') character,
  // try again without the underscore.
  if (NameStr[0] == '_') {
    Ptr = sys::DynamicLibrary::SearchForAddressOfSymbol(NameStr+1);
    if (Ptr) return Ptr;
  }

  if (AbortOnFailure)
    report_fatal_error("Program used external function '" + Name +
                      "' which could not be resolved!");
  return 0;
}

SectionMemoryManager::~SectionMemoryManager() {
  for (unsigned i = 0, e = AllocatedCodeMem.size(); i != e; ++i)
    sys::Memory::ReleaseRWX(AllocatedCodeMem[i]);
  for (unsigned i = 0, e = AllocatedDataMem.size(); i != e; ++i)
    free(AllocatedDataMem[i].base());
}

} // namespace llvm
