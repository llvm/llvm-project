//===--------- RegisterEHFrames.cpp - Register EH frame sections ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"

#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace llvm {
namespace orc {

#if defined(HAVE_REGISTER_FRAME) && defined(HAVE_DEREGISTER_FRAME) &&          \
    !defined(__SEH__) && !defined(__USING_SJLJ_EXCEPTIONS__)

extern "C" void __register_frame(const void *);
extern "C" void __deregister_frame(const void *);

#if defined(__ELF__) && LLVM_HAS_ATTRIBUTE_WEAK
#define LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION 1
// LLVM's libunwind exports section-level registration entry points alongside
// __unw_add_dynamic_fde, whereas its __register_frame only accepts a single FDE
// rather than a null-terminated .eh_frame section. Detect them dynamically via
// weak references (matching compiler-rt's elfnix_platform) and fall back to
// __register_frame (e.g. libgcc) otherwise.
extern "C" LLVM_ATTRIBUTE_WEAK void
__unw_add_dynamic_eh_frame_section(uintptr_t);
extern "C" LLVM_ATTRIBUTE_WEAK void
__unw_remove_dynamic_eh_frame_section(uintptr_t);
#else
#define LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION 0
#endif

Error registerFrameWrapper(const void *P) {
#if LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION
  if (&__unw_add_dynamic_eh_frame_section &&
      &__unw_remove_dynamic_eh_frame_section) {
    __unw_add_dynamic_eh_frame_section(reinterpret_cast<uintptr_t>(P));
    return Error::success();
  }
#endif
  __register_frame(P);
  return Error::success();
}

Error deregisterFrameWrapper(const void *P) {
#if LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION
  if (&__unw_add_dynamic_eh_frame_section &&
      &__unw_remove_dynamic_eh_frame_section) {
    __unw_remove_dynamic_eh_frame_section(reinterpret_cast<uintptr_t>(P));
    return Error::success();
  }
#endif
  __deregister_frame(P);
  return Error::success();
}

#else
#define LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION 0

// The building compiler does not have __(de)register_frame but
// it may be found at runtime in a dynamically-loaded library.
// For example, this happens when building LLVM with Visual C++
// but using the MingW runtime.
static Error registerFrameWrapper(const void *P) {
  static void((*RegisterFrame)(const void *)) = 0;

  if (!RegisterFrame)
    *(void **)&RegisterFrame =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("__register_frame");

  if (RegisterFrame) {
    RegisterFrame(P);
    return Error::success();
  }

  return make_error<StringError>("could not register eh-frame: "
                                 "__register_frame function not found",
                                 inconvertibleErrorCode());
}

static Error deregisterFrameWrapper(const void *P) {
  static void((*DeregisterFrame)(const void *)) = 0;

  if (!DeregisterFrame)
    *(void **)&DeregisterFrame =
        llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
            "__deregister_frame");

  if (DeregisterFrame) {
    DeregisterFrame(P);
    return Error::success();
  }

  return make_error<StringError>("could not deregister eh-frame: "
                                 "__deregister_frame function not found",
                                 inconvertibleErrorCode());
}
#endif

#if (defined(HAVE_UNW_ADD_DYNAMIC_FDE) || defined(__APPLE__)) &&               \
    !LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION

template <typename HandleFDEFn>
Error walkLibunwindEHFrameSection(const char *const SectionStart,
                                  size_t SectionSize, HandleFDEFn HandleFDE) {
  const char *CurCFIRecord = SectionStart;
  const char *End = SectionStart + SectionSize;
  uint64_t Size = *reinterpret_cast<const uint32_t *>(CurCFIRecord);

  while (CurCFIRecord != End && Size != 0) {
    const char *OffsetField = CurCFIRecord + (Size == 0xffffffff ? 12 : 4);
    if (Size == 0xffffffff)
      Size = *reinterpret_cast<const uint64_t *>(CurCFIRecord + 4) + 12;
    else
      Size += 4;
    uint32_t Offset = *reinterpret_cast<const uint32_t *>(OffsetField);

    LLVM_DEBUG({
      dbgs() << "Registering eh-frame section:\n";
      dbgs() << "Processing " << (Offset ? "FDE" : "CIE") << " @"
             << (void *)CurCFIRecord << ": [";
      for (unsigned I = 0; I < Size; ++I)
        dbgs() << format(" 0x%02" PRIx8, *(CurCFIRecord + I));
      dbgs() << " ]\n";
    });

    if (Offset != 0)
      if (auto Err = HandleFDE(CurCFIRecord))
        return Err;

    CurCFIRecord += Size;

    Size = *reinterpret_cast<const uint32_t *>(CurCFIRecord);
  }

  return Error::success();
}

#endif // (HAVE_UNW_ADD_DYNAMIC_FDE || __APPLE__) &&
       // !LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION

Error registerEHFrameSection(const void *EHFrameSectionAddr,
                             size_t EHFrameSectionSize) {
  /* libgcc and libunwind __register_frame behave differently. On ELF targets
   * with weak symbol support, registerFrameWrapper dynamically prefers
   * libunwind's section-level entry points; elsewhere we use
   * HAVE_UNW_ADD_DYNAMIC_FDE or __APPLE__ to detect per-FDE registration. */
#if (defined(HAVE_UNW_ADD_DYNAMIC_FDE) || defined(__APPLE__)) &&               \
    !LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION
  // With libunwind, __register_frame has to be called for each FDE entry.
  return walkLibunwindEHFrameSection(
      static_cast<const char *>(EHFrameSectionAddr), EHFrameSectionSize,
      registerFrameWrapper);
#else
  // With libgcc, __register_frame takes a single argument:
  // a pointer to the start of the .eh_frame section.

  // How can it find the end? Because crtendS.o is linked
  // in and it has an .eh_frame section with four zero chars.
  (void)EHFrameSectionSize;
  return registerFrameWrapper(EHFrameSectionAddr);
#endif
}

Error deregisterEHFrameSection(const void *EHFrameSectionAddr,
                               size_t EHFrameSectionSize) {
#if (defined(HAVE_UNW_ADD_DYNAMIC_FDE) || defined(__APPLE__)) &&               \
    !LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION
  return walkLibunwindEHFrameSection(
      static_cast<const char *>(EHFrameSectionAddr), EHFrameSectionSize,
      deregisterFrameWrapper);
#else
  (void)EHFrameSectionSize;
  return deregisterFrameWrapper(EHFrameSectionAddr);
#endif
}

#undef LLVM_ORC_DYNAMIC_ELF_EH_FRAME_REGISTRATION

} // end namespace orc
} // end namespace llvm

static Error registerEHFrameWrapper(ExecutorAddrRange EHFrame) {
  return llvm::orc::registerEHFrameSection(EHFrame.Start.toPtr<const void *>(),
                                           EHFrame.size());
}

static Error deregisterEHFrameWrapper(ExecutorAddrRange EHFrame) {
  return llvm::orc::deregisterEHFrameSection(
      EHFrame.Start.toPtr<const void *>(), EHFrame.size());
}

extern "C" orc::shared::CWrapperFunctionBuffer
llvm_orc_registerEHFrameSectionAllocAction(const char *ArgData,
                                           size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddrRange)>::handle(
             ArgData, ArgSize, registerEHFrameWrapper)
      .release();
}

extern "C" orc::shared::CWrapperFunctionBuffer
llvm_orc_deregisterEHFrameSectionAllocAction(const char *ArgData,
                                             size_t ArgSize) {
  return WrapperFunction<SPSError(SPSExecutorAddrRange)>::handle(
             ArgData, ArgSize, deregisterEHFrameWrapper)
      .release();
}
