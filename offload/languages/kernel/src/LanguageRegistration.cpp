//===-- LanguageRegistration.cpp - Language registration API --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LanguageRegistration.h"
#include "OffloadAPI.h"
#include "State.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <cstring>
#include <inttypes.h>

using RuntimeState = llvm::offload::StateTy;
using ThreadState = llvm::offload::ThreadStateTy;

/// Hidden, but exported, Registration API
///{
extern "C" {

void __llvmRegisterFunction(const char *Binary, const char *KernelID,
                            char *KernelName, const char *KernelName1, int,
                            uint3 *, uint3 *, dim3 *, dim3 *, int *) {
  RuntimeState &State = RuntimeState::get();
  ol_symbol_handle_t Kernel;
  ol_program_handle_t Program = State.getProgram(Binary);
  ol_result_t Result = olGetSymbol(
      Program, KernelName, ol_symbol_kind_t::OL_SYMBOL_KIND_KERNEL, &Kernel);
  CHECK_FATAL(Result, "Failed to get kernel symbol for " << KernelName);
  State.registerKernel(KernelID, Kernel);
}

void __llvmRegisterVar(void **, char *, char *, const char *, int, int, int,
                       int) {
  llvm::errs() << "RegisterVar is not implemented!" << "\n";
}

void __llvmRegisterManagedVar(void **, char *, char *, const char *, size_t,
                              unsigned) {
  llvm::errs() << "RegisterManagedVar is not implemented!" << "\n";
}

void __llvmRegisterSurface(void **, const struct surfaceReference *,
                           const void **, const char *, int, int) {
  llvm::errs() << "RegisterSurface is not implemented!" << "\n";
}

void __llvmRegisterTexture(void **, const struct textureReference *,
                           const void **, const char *, int, int, int) {
  llvm::errs() << "RegisterTexture is not implemented!" << "\n";
}

/// This struct is a record of the device image information
struct __tgt_device_image {
  void *ImageStart; // Pointer to the target code start
  void *ImageEnd;   // Pointer to the target code end
  llvm::offloading::EntryTy
      *EntriesBegin; // Begin of table with all target entries
  llvm::offloading::EntryTy *EntriesEnd; // End of table (non inclusive)
};

/// This struct is a record of all the host code that may be offloaded to a
/// target.
struct __tgt_bin_desc {
  int32_t NumDeviceImages;          // Number of device types supported
  __tgt_device_image *DeviceImages; // Array of device images (1 per dev. type)
  llvm::offloading::EntryTy
      *HostEntriesBegin; // Begin of table with all host entries
  llvm::offloading::EntryTy *HostEntriesEnd; // End of table (non inclusive)
};

void __tgt_register_lib(__tgt_bin_desc *Desc) {
  // TODO: For each device, lazily.
  RuntimeState &State = RuntimeState::get();
  ThreadState &Thread = ThreadState::get();
  ol_device_handle_t Device = Thread.getDefaultDevice();

  for (int32_t I = 0, E = Desc->NumDeviceImages; I < E; ++I) {
    ol_program_handle_t Program = nullptr;

    __tgt_device_image &DeviceImage = Desc->DeviceImages[I];
    void *ProgramData = DeviceImage.ImageStart;
    size_t ProgramSize =
        (char *)DeviceImage.ImageEnd - (char *)DeviceImage.ImageStart;
    ol_result_t Result =
        olCreateProgram(Device, ProgramData, ProgramSize, &Program);

    if (Result && Result->Code) {
      fprintf(stderr, "Failed to register device code (%i): %s\n", Result->Code,
              Result->Details);
      abort();
    }

    State.registerProgram(DeviceImage.ImageStart, Program);

    for (auto *Entry = DeviceImage.EntriesBegin;
         Entry != DeviceImage.EntriesEnd; ++Entry) {
      if (!Entry->Size && !Entry->Flags)
        __llvmRegisterFunction((const char *)DeviceImage.ImageStart,
                               (const char *)Entry->Address, Entry->SymbolName,
                               Entry->SymbolName, 0, nullptr, nullptr, nullptr,
                               nullptr, nullptr);
    }
  }
}

void __tgt_unregister_lib(__tgt_bin_desc *Desc) {
  RuntimeState *State = RuntimeState::tryGet();
  if (!State)
    return;

  for (int32_t I = 0, E = Desc->NumDeviceImages; I < E; ++I) {
    __tgt_device_image &DeviceImage = Desc->DeviceImages[I];
    for (auto *Entry = DeviceImage.EntriesBegin;
         Entry != DeviceImage.EntriesEnd; ++Entry) {
      if (!Entry->Size && !Entry->Flags)
        State->unregisterKernel((const char *)Entry->Address);
    }

    if (ol_program_handle_t Program =
            State->unregisterProgram(DeviceImage.ImageStart))
      olDestroyProgram(Program);
  }
}
}
///}
