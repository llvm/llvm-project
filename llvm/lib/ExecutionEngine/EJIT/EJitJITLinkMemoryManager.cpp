//===-- EJitJITLinkMemoryManager.cpp - Embedded Memory Manager -------------===//
//
// This is a stub implementation. The full embedded slab memory manager
// requires implementing the InFlightAlloc subclass, which will be completed
// in a follow-up change. For now, the OrcEngine uses the default LLJIT
// memory manager.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/EJIT/EJitJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Support/Error.h"
#include <sys/mman.h>

using namespace llvm;
using namespace llvm::ejit;

static void *allocateSlab(size_t size) {
  return mmap(nullptr, size, PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
}

EJitJITLinkMemoryManager::EJitJITLinkMemoryManager(size_t maxCodeSize,
                                                   size_t maxDataSize) {
  codeSlab_.totalSize = maxCodeSize;
  dataSlab_.totalSize = maxDataSize;
  codeSlab_.baseAddr = allocateSlab(maxCodeSize);
  dataSlab_.baseAddr = allocateSlab(maxDataSize);
}

EJitJITLinkMemoryManager::~EJitJITLinkMemoryManager() {
  if (codeSlab_.baseAddr)
    munmap(codeSlab_.baseAddr, codeSlab_.totalSize);
  if (dataSlab_.baseAddr)
    munmap(dataSlab_.baseAddr, dataSlab_.totalSize);
}

void EJitJITLinkMemoryManager::allocate(
    const jitlink::JITLinkDylib *JD,
    jitlink::LinkGraph &G,
    OnAllocatedFunction OnAllocated) {
  // TODO: Full implementation with InFlightAlloc subclass
  OnAllocated(make_error<StringError>(
      "EJitJITLinkMemoryManager not fully implemented yet",
      inconvertibleErrorCode()));
}

void EJitJITLinkMemoryManager::deallocate(
    std::vector<FinalizedAlloc> Allocs,
    OnDeallocatedFunction OnDeallocated) {
  OnDeallocated(Error::success());
}

size_t EJitJITLinkMemoryManager::getCurrentCodeUsage() const {
  return codeSlab_.usedSize;
}

size_t EJitJITLinkMemoryManager::getCurrentDataUsage() const {
  return dataSlab_.usedSize;
}

void EJitJITLinkMemoryManager::reset() {
  std::lock_guard<std::mutex> lockC(codeSlab_.mutex);
  std::lock_guard<std::mutex> lockD(dataSlab_.mutex);
  codeSlab_.usedSize = 0;
  dataSlab_.usedSize = 0;
}
