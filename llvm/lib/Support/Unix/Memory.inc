//===- Unix/Memory.cpp - Generic UNIX System Configuration ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines some functions for various memory management utilities.
//
//===----------------------------------------------------------------------===//

#include "Unix.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Valgrind.h"

#ifdef HAVE_SYS_MMAN_H
#include <sys/mman.h>
#endif

#ifdef __linux__
#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/vfs.h>

#include <fstream>

// We use a weak reference to avoid requiring to link against libselinux just
// for this but if users do so anyway, we'll make use of it.
extern "C" int is_selinux_enabled(void) LLVM_ATTRIBUTE_WEAK;
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#ifdef __Fuchsia__
#include <zircon/syscalls.h>
#endif

#if defined(__APPLE__)
extern "C" void sys_icache_invalidate(const void *Addr, size_t len);
#else
extern "C" void __clear_cache(void *, void *);
#endif

static int getPosixProtectionFlags(unsigned Flags) {
  switch (Flags & llvm::sys::Memory::MF_RWE_MASK) {
  case llvm::sys::Memory::MF_READ:
    return PROT_READ;
  case llvm::sys::Memory::MF_WRITE:
    return PROT_WRITE;
  case llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE:
    return PROT_READ | PROT_WRITE;
  case llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_EXEC:
    return PROT_READ | PROT_EXEC;
  case llvm::sys::Memory::MF_READ | llvm::sys::Memory::MF_WRITE |
      llvm::sys::Memory::MF_EXEC:
    return PROT_READ | PROT_WRITE | PROT_EXEC;
  case llvm::sys::Memory::MF_EXEC:
#if defined(__FreeBSD__) || defined(__powerpc__)
    // On PowerPC, having an executable page that has no read permission
    // can have unintended consequences.  The function InvalidateInstruction-
    // Cache uses instructions dcbf and icbi, both of which are treated by
    // the processor as loads.  If the page has no read permissions,
    // executing these instructions will result in a segmentation fault.
    return PROT_READ | PROT_EXEC;
#else
    return PROT_EXEC;
#endif
  default:
    llvm_unreachable("Illegal memory protection flag specified!");
  }
  // Provide a default return value as required by some compilers.
  return PROT_NONE;
}

#if defined(__linux__)
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U

// Assume if MFD_CLOEXEC is defined, so is the libc function
static inline int memfd_create(const char *name, unsigned int flags) {
#ifdef SYS_memfd_create
  return syscall(SYS_memfd_create, name, flags);
#else
  errno = ENOSYS;
  return -1;
#endif
}
#endif // MFD_CLOEXEC
#ifndef MFD_EXEC
#define MFD_EXEC 0x0010U
#endif
#endif

namespace llvm {
namespace sys {

MemoryBlock Memory::allocateMappedMemory(size_t NumBytes,
                                         const MemoryBlock *const NearBlock,
                                         unsigned PFlags, std::error_code &EC) {
  EC = std::error_code();
  if (NumBytes == 0)
    return MemoryBlock();

  // On platforms that have it, we can use MAP_ANON to get a memory-mapped
  // page without file backing, but we need a fallback of opening /dev/zero
  // for strictly POSIX platforms instead.
  int fd;
#if defined(MAP_ANON)
  fd = -1;
#else
  fd = open("/dev/zero", O_RDWR);
  if (fd == -1) {
    EC = errnoAsErrorCode();
    return MemoryBlock();
  }
#endif

  int MMFlags = MAP_PRIVATE;
#if defined(MAP_ANON)
  MMFlags |= MAP_ANON;
#endif
  int Protect = getPosixProtectionFlags(PFlags);

#if defined(__NetBSD__) && defined(PROT_MPROTECT)
  Protect |= PROT_MPROTECT(PROT_READ | PROT_WRITE | PROT_EXEC);
#endif

  // Use any near hint and the page size to set a page-aligned starting address
  uintptr_t Start = NearBlock ? reinterpret_cast<uintptr_t>(NearBlock->base()) +
                                    NearBlock->allocatedSize()
                              : 0;
  static const size_t PageSize = Process::getPageSizeEstimate();
  const size_t NumPages = (NumBytes + PageSize - 1) / PageSize;

  if (Start && Start % PageSize)
    Start += PageSize - Start % PageSize;

  // FIXME: Handle huge page requests (MF_HUGE_HINT).
  void *Addr = ::mmap(reinterpret_cast<void *>(Start), PageSize * NumPages,
                      Protect, MMFlags, fd, 0);
  if (Addr == MAP_FAILED) {
    if (NearBlock) { // Try again without a near hint
#if !defined(MAP_ANON)
      close(fd);
#endif
      return allocateMappedMemory(NumBytes, nullptr, PFlags, EC);
    }

    EC = errnoAsErrorCode();
#if !defined(MAP_ANON)
    close(fd);
#endif
    return MemoryBlock();
  }

#if !defined(MAP_ANON)
  close(fd);
#endif

  MemoryBlock Result;
  Result.Address = Addr;
  Result.AllocatedSize = PageSize * NumPages;
  Result.Flags = PFlags;

  // Rely on protectMappedMemory to invalidate instruction cache.
  if (PFlags & MF_EXEC) {
    EC = Memory::protectMappedMemory(Result, PFlags);
    if (EC != std::error_code())
      return MemoryBlock();
  }

  return Result;
}

std::error_code Memory::releaseMappedMemory(MemoryBlock &M) {
  if (M.Address == nullptr || M.AllocatedSize == 0)
    return std::error_code();

  if (0 != ::munmap(M.Address, M.AllocatedSize))
    return errnoAsErrorCode();

  M.Address = nullptr;
  M.AllocatedSize = 0;

  return std::error_code();
}

std::error_code Memory::protectMappedMemory(const MemoryBlock &M,
                                            unsigned Flags) {
  static const Align PageSize = Align(Process::getPageSizeEstimate());
  if (M.Address == nullptr || M.AllocatedSize == 0)
    return std::error_code();

  if (!Flags)
    return std::error_code(EINVAL, std::generic_category());

  int Protect = getPosixProtectionFlags(Flags);
  uintptr_t Start =
      alignAddr((const uint8_t *)M.Address - PageSize.value() + 1, PageSize);
  uintptr_t End =
      alignAddr((const uint8_t *)M.Address + M.AllocatedSize, PageSize);

  bool InvalidateCache = (Flags & MF_EXEC);
  bool SkipMprotect = false;

#if defined(__linux__)
  // Check for cases where the EXEC protection flag changes and a possible
  // strict W^X policy cannot be bypassed via mprotect() alone, e.g. under
  // PaX's MPROTECT or SELinux's deny_execmem.
  //
  // To support such systems, we need to create a fresh mapping with the
  // target protection flags.
  if ((M.Flags ^ Flags) & MF_EXEC && execProtectionChangeNeedsNewMapping()) {
    // The initial version of memfd_create() created executable fds by default.
    // Since Linux v6.3 the default behaviour depends on a sysctl setting
    // 'vm.memfd_noexec', requiring users to explicitly request the executable
    // status by passing MFD_EXEC in the flags. This flag will be rejected by
    // earlier kernels with EINVAL. Simply retry without the flag in this case,
    // as mappings will be executable by default on these kernels.
    static unsigned int flags = MFD_CLOEXEC | MFD_EXEC;
    int fd = memfd_create("llvm", flags);
    if (fd < 0 && errno == EINVAL && (flags & MFD_EXEC)) {
      flags &= ~MFD_EXEC;
      fd = memfd_create("llvm", flags);
    }
    if (fd < 0)
      return errnoAsErrorCode();

    raw_fd_ostream fdos(fd, /*shouldClose=*/true, /*unbuffered=*/true,
                        /*simpleStream=*/true);

    fdos.write(reinterpret_cast<char *>(Start), End - Start);
    if (auto ec = fdos.error())
      return ec;

    void *addr = ::mmap(reinterpret_cast<void *>(Start), End - Start, Protect,
                        MAP_PRIVATE | MAP_FIXED, fd, 0);
    if (addr == MAP_FAILED)
      return errnoAsErrorCode();

    // We created a new mapping with the final protection bits, therefore
    // don't need to call mprotect() with the very same flags again -- unless
    // we have to toggle PROT_READ for ARM.
    SkipMprotect = true;
  }
#endif

#if defined(__arm__) || defined(__aarch64__)
  // Certain ARM implementations treat icache clear instruction as a memory
  // read, and CPU segfaults on trying to clear cache on !PROT_READ page.
  // Therefore we need to temporarily add PROT_READ for the sake of flushing the
  // instruction caches.
  if (InvalidateCache && !(Protect & PROT_READ)) {
    int Result = ::mprotect((void *)Start, End - Start, Protect | PROT_READ);
    if (Result != 0)
      return errnoAsErrorCode();

    Memory::InvalidateInstructionCache(M.Address, M.AllocatedSize);
    InvalidateCache = false;
    SkipMprotect = false;
  }
#endif

  if (!SkipMprotect) {
    int Result = ::mprotect((void *)Start, End - Start, Protect);

    if (Result != 0)
      return errnoAsErrorCode();
  }

  if (InvalidateCache)
    Memory::InvalidateInstructionCache(M.Address, M.AllocatedSize);

  return std::error_code();
}

/// InvalidateInstructionCache - Before the JIT can run a block of code
/// that has been emitted it must invalidate the instruction cache on some
/// platforms.
void Memory::InvalidateInstructionCache(const void *Addr, size_t Len) {

// icache invalidation for PPC and ARM.
#if defined(__APPLE__)

#if (defined(__powerpc__) || defined(__arm__) || defined(__arm64__))
  sys_icache_invalidate(const_cast<void *>(Addr), Len);
#endif

#elif defined(__Fuchsia__)

  zx_status_t Status = zx_cache_flush(Addr, Len, ZX_CACHE_FLUSH_INSN);
  assert(Status == ZX_OK && "cannot invalidate instruction cache");

#else

#if defined(__powerpc__) && defined(__GNUC__)
  const size_t LineSize = 32;

  const intptr_t Mask = ~(LineSize - 1);
  const intptr_t StartLine = ((intptr_t)Addr) & Mask;
  const intptr_t EndLine = ((intptr_t)Addr + Len + LineSize - 1) & Mask;

  for (intptr_t Line = StartLine; Line < EndLine; Line += LineSize)
    asm volatile("dcbf 0, %0" : : "r"(Line));
  asm volatile("sync");

  for (intptr_t Line = StartLine; Line < EndLine; Line += LineSize)
    asm volatile("icbi 0, %0" : : "r"(Line));
  asm volatile("isync");
#elif (defined(__arm__) || defined(__aarch64__) || defined(__loongarch__) ||   \
       defined(__mips__)) &&                                                   \
    defined(__GNUC__)
  // FIXME: Can we safely always call this for __GNUC__ everywhere?
  const char *Start = static_cast<const char *>(Addr);
  const char *End = Start + Len;
  __clear_cache(const_cast<char *>(Start), const_cast<char *>(End));
#endif

#endif // end apple

  ValgrindDiscardTranslations(Addr, Len);
}

#if defined(__linux__)
static int checkPaX() {
  // Try to get the status from /proc/self/status, looking for PaX flags.
  if (auto file = MemoryBuffer::getFileAsStream("/proc/self/status")) {
    auto pax_flags =
        (*file)->getBuffer().rsplit("PaX:").second.split('\n').first.trim();
    if (!pax_flags.empty())
      // 'M' indicates MPROTECT is enabled
      return pax_flags.find('M') != StringRef::npos;
  }

  return -1;
}

// see linux/security/selinux/selinuxfs.c
#ifndef SEL_INO_MASK
#define SEL_INO_MASK 0x00ffffff
#endif
#ifndef SEL_CLASS_INO_OFFSET
#define SEL_CLASS_INO_OFFSET 0x04000000
#endif
#ifndef SEL_VEC_MAX
#define SEL_VEC_MAX 32
#endif
#ifndef SELINUX_MAGIC
#define SELINUX_MAGIC 0xf97cff8c
#endif

#define SEL_MNT "/sys/fs/selinux"

static int SELinuxEnabled() {
  if (is_selinux_enabled)
    return is_selinux_enabled();

  struct statfs fsbuf;
  if (statfs(SEL_MNT, &fsbuf) < 0)
    return -1;

  return (unsigned int)fsbuf.f_type == (unsigned int)SELINUX_MAGIC;
}

// Test if the current context is allowed to "execmem"
static int checkSELinux() {
  int se_status = SELinuxEnabled();
  if (se_status <= 0)
    return se_status;

  auto context = MemoryBuffer::getFileAsStream("/proc/self/attr/current");
  if (!context)
    return -1;

  // The inode encodes class and permission bits
  struct stat sbuf;
  if (stat(SEL_MNT "/class/process/perms/execmem", &sbuf))
    return -1;

  if (!(sbuf.st_ino & SEL_CLASS_INO_OFFSET))
    return -1;

  int clazz = (sbuf.st_ino & SEL_INO_MASK) / (SEL_VEC_MAX + 1);
  int xperm = (sbuf.st_ino & SEL_INO_MASK) % (SEL_VEC_MAX + 1) - 1;
  if (xperm < 0 || xperm > 31)
    return -1;

  unsigned int execmem = 1U << xperm;

  std::fstream access(SEL_MNT "/access", std::ios::in | std::ios::out);
  if (!access)
    return -1;

  access << (*context)->getBuffer().str().c_str() << " "
         << (*context)->getBuffer().str().c_str() << " " << clazz << " "
         << std::hex << execmem;
  access.flush();
  if (access.bad())
    return -1;

  unsigned int lo, hi;
  access >> lo >> hi;
  if (access.bad())
    return -1;

  uint64_t allowed = (uint64_t)hi << 32 | lo;
  return !(allowed & execmem);
}

static bool isPermissionError(int err) {
  // PaX uses EPERM, SELinux uses EACCES
  return err == EPERM || err == EACCES;
}

static int checkSyscall() {
  // Create a temporary writable mapping and try to make it executable.  If
  // this fails, test 'errno' to ensure it failed because we were not allowed
  // to create such a mapping and not because of some transient error.
  size_t size = Process::getPageSizeEstimate();
  void *addr = ::mmap(NULL, size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED)
    // Must be low on memory or have too many mappings already, not much we can
    // do here.
    return 0;

  int status;
  if (::mprotect(addr, size, PROT_READ | PROT_EXEC) < 0)
    status = isPermissionError(errno);
  else
    status = 0;
  ::munmap(addr, size);

  return status;
}

bool Memory::execProtectionChangeNeedsNewMapping() {
  static int status = -1;
  int pax, selinux;

  if (status != -1)
    return status;

  // check PaX first, easiest
  if ((pax = checkPaX()) == 1)
    return (status = 1);

  // SELinux may still deny even if PaX allows
  if ((selinux = checkSELinux()) == 1)
    return (status = 1);

  // if either has a verdict, it's governing the behaviour
  if (pax == 0 || selinux == 0)
    return (status = 0);

  // inconclusive results, do a runtime check
  status = checkSyscall();

  return status;
}
#else
bool Memory::execProtectionChangeNeedsNewMapping() { return false; }
#endif

} // namespace sys
} // namespace llvm
