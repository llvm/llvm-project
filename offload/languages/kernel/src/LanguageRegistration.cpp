//===---- LanguageRegistration.h - Language (CUDA/HIP) registration api ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LanguageRegistration.h"
#include "OffloadAPI.h"
#include "RuntimeAPI.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Support/Error.h"
#include <cstdio>
#include <inttypes.h>

namespace language_registration = llvm::offload::kernel;

typedef struct __attribute__((__packed__)) {
  uint32_t Magic;
  uint16_t Version;
  uint16_t HeaderSize;
  uint64_t FatSize;
} CudaFatbinHeader;

// Inspired by
// https://github.com/n-eiling/cuda-fatbin-decompression/blob/master/fatbin-decompress.h
typedef struct __attribute__((__packed__)) {
  uint16_t Kind;
  uint16_t Unknown1;
  uint32_t HeaderSize;
  uint64_t Size;
  uint32_t CompressedSize;
  uint32_t Unknown2;
  uint16_t Minor;
  uint16_t Major;
  uint32_t Arch;
  uint32_t ObjNameOffset;
  uint32_t ObjNameLen;
  uint64_t Flags;
  uint64_t Zero;
  uint64_t DecompressedSize;
} CudaFatbinTextHeader;

// HIP uses this format:
// https://clang.llvm.org/docs/ClangOffloadBundler.html#bundled-binary-file-layout
typedef struct __attribute__((__packed__)) {
  char Magic[24];
  uint64_t NumBundles;
} HipFatbinHeader;

typedef struct __attribute__((__packed__)) {
  uint64_t BundleOffset;
  uint64_t BundleSize;
  uint64_t IdLength;
  char IdString[];
} HipFatbinBundleEntry;

static void readTUFatbin(const char *Binary, const FatbinWrapperTy *FW) {
  ol_device_handle_t Device = language_registration::getDefaultDevice();

  const CudaFatbinHeader *Header =
      reinterpret_cast<const CudaFatbinHeader *>(FW->Data);
  size_t HeaderSize = static_cast<size_t>(Header->HeaderSize); // Usually 16
  size_t FatbinSize = static_cast<size_t>(Header->FatSize);

  const void *ProgramData = nullptr;
  size_t ProgramSize = 0;
  uint32_t ProgramArch = 0;

  const char *ReadPosition = FW->Data + HeaderSize;
  while (ReadPosition < (FW->Data + FatbinSize)) {
    const CudaFatbinTextHeader *TextHeader =
        reinterpret_cast<const CudaFatbinTextHeader *>(ReadPosition);
    size_t TextHeaderSize =
        static_cast<size_t>(TextHeader->HeaderSize); // Usually 64
    size_t CubinSize = static_cast<size_t>(TextHeader->Size);
    const void *CubinData =
        static_cast<const char *>(ReadPosition + TextHeaderSize);

    uint32_t Arch = TextHeader->Arch;
    bool IsCompatible = false;
    olIsValidBinary(Device, CubinData, CubinSize, &IsCompatible);
    if (!IsCompatible) {
      fprintf(stderr, "Device is not compatible with image.");
      abort();
    }

    if (Arch > ProgramArch) {
      ProgramData = CubinData;
      ProgramSize = CubinSize;
      ProgramArch = Arch;
    }

    ReadPosition += TextHeaderSize + CubinSize;
  }

  if (ProgramData == nullptr) {
    fprintf(stderr, "Failed to find compatible binary\n");
    abort();
  }

  ol_program_handle_t Program = nullptr;

  ol_result_t Result =
      olCreateProgram(Device, ProgramData, ProgramSize, &Program);

  if (Result && Result->Code) {
    fprintf(stderr, "Failed to register device code (%i): %s\n", Result->Code,
            Result->Details);
    abort();
  }

  language_registration::registerProgram(Binary, Program);
}

static void readHIPFatbinEntries(const char *Binary, const char *HIPFatbinPtr) {
  ol_device_handle_t Device = language_registration::getDefaultDevice();

  const char *CurrentReadPosition = HIPFatbinPtr;

  const HipFatbinHeader *Header =
      reinterpret_cast<const HipFatbinHeader *>(CurrentReadPosition);
  CurrentReadPosition += sizeof(HipFatbinHeader);

  uint64_t NumBundles = Header->NumBundles;

  const void *ProgramData = nullptr;
  size_t ProgramSize = 0;
  uint64_t ProgramIdLength = 0;
  const char *ProgramIdString = nullptr;

  for (uint64_t BundleId = 0; BundleId < NumBundles; ++BundleId) {
    const HipFatbinBundleEntry *BundleEntry =
        reinterpret_cast<const HipFatbinBundleEntry *>(CurrentReadPosition);

    uint64_t BundleOffset = BundleEntry->BundleOffset;
    uint64_t BundleSize = BundleEntry->BundleSize;
    const char *BundleIdString = BundleEntry->IdString;
    uint64_t BundleIdLength = BundleEntry->IdLength;

    // Advance by the size of the entry including the ID string
    CurrentReadPosition += sizeof(HipFatbinBundleEntry) + BundleIdLength;

    if (!BundleSize) {
      continue;
    }

    bool IsCompatible = false;
    olIsValidBinary(Device, HIPFatbinPtr + BundleOffset, BundleSize,
                    &IsCompatible);

    if (!IsCompatible) {
      fprintf(stderr, "Device is not compatible with image.");
      abort();
    }

    llvm::StringRef CurrentBundleId(ProgramIdString, ProgramIdLength);
    llvm::StringRef NewBundleId(BundleIdString, BundleIdLength);
    if (NewBundleId.compare(CurrentBundleId) > 0) {
      ProgramData = HIPFatbinPtr + BundleOffset;
      ProgramSize = BundleSize;
      ProgramIdLength = BundleIdLength;
      ProgramIdString = BundleIdString;
    }
  }

  if (ProgramData == nullptr) {
    fprintf(stderr, "Failed to find compatible binary\n");
    abort();
  }

  ol_program_handle_t Program = nullptr;
  ol_result_t Result =
      olCreateProgram(Device, ProgramData, ProgramSize, &Program);
  if (Result && Result->Code) {
    fprintf(stderr, "Failed to register device code (%i): %s\n", Result->Code,
            Result->Details);
    abort();
  }

  language_registration::registerProgram(Binary, Program);
}

/// Hidden, but exported, Registration API
///{
extern "C" {

void __llvmRegisterFunction(const char *Binary, const char *KernelID,
                            char *KernelName, const char *KernelName1, int,
                            uint3 *, uint3 *, dim3 *, dim3 *, int *) {
  ol_symbol_handle_t Kernel;
  ol_program_handle_t Program = language_registration::getProgram(Binary);
  ol_result_t Result = olGetSymbol(
      Program, KernelName, ol_symbol_kind_t::OL_SYMBOL_KIND_KERNEL, &Kernel);
  if (Result && Result->Code) {
    fprintf(stderr, "Failed to register kernel (%i): %s\n", Result->Code,
            Result->Details);
    abort();
  }

  language_registration::registerKernel(KernelID, Kernel);
}

const char *__llvmRegisterFatBinary(const char *Binary) {
  const auto *FW = reinterpret_cast<const FatbinWrapperTy *>(Binary);
  if (FW->Magic == 0x466243b1) {
    readTUFatbin(Binary, FW);
  } else if (FW->Magic == 0x48495046) {
    if (!memcmp(FW->Data, HIP_FATBIN_MAGIC_STR, HIP_FATBIN_MAGIC_STR_LEN))
      readHIPFatbinEntries(Binary, FW->Data);
    else
      readTUFatbin(Binary, FW);
  } else {
    fprintf(stderr, "Unknown fatbin format");
  }

  return Binary;
}

void __llvmUnregisterFatBinary(void *Handle) {
  if (ol_program_handle_t Program =
          language_registration::unregisterProgram(Handle))
    olDestroyProgram(Program);
}

void __llvmRegisterVar(void **, char *, char *, const char *, int, int, int,
                       int) {
  fprintf(stderr, "RegisterVar is not implemented!");
}

void __llvmRegisterManagedVar(void **, char *, char *, const char *, size_t,
                              unsigned) {
  fprintf(stderr, "RegisterManagedVar is not implemented!");
}

void __llvmRegisterSurface(void **, const struct surfaceReference *,
                           const void **, const char *, int, int) {
  fprintf(stderr, "RegisterSurface is not implemented!");
}

void __llvmRegisterTexture(void **, const struct textureReference *,
                           const void **, const char *, int, int, int) {
  fprintf(stderr, "RegisterTexture is not implemented!");
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
  ol_device_handle_t Device = language_registration::getDefaultDevice();

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

    language_registration::registerProgram(DeviceImage.ImageStart, Program);

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
  for (int32_t I = 0, E = Desc->NumDeviceImages; I < E; ++I) {
    __tgt_device_image &DeviceImage = Desc->DeviceImages[I];
    for (auto *Entry = DeviceImage.EntriesBegin;
         Entry != DeviceImage.EntriesEnd; ++Entry) {
      if (!Entry->Size && !Entry->Flags)
        language_registration::unregisterKernel((const char *)Entry->Address);
    }

    if (ol_program_handle_t Program =
            language_registration::unregisterProgram(DeviceImage.ImageStart))
      olDestroyProgram(Program);
  }
}
}
///}
