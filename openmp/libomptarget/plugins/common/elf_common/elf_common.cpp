//===-- elf_common.cpp - Common ELF functionality -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common ELF functionality for target plugins.
//
//===----------------------------------------------------------------------===//
#include "elf_common.h"
#include "Debug.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

#ifndef TARGET_NAME
#define TARGET_NAME ELF Common
#endif
#define DEBUG_PREFIX "TARGET " GETNAME(TARGET_NAME)

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

/// If the given range of bytes [\p BytesBegin, \p BytesEnd) represents
/// a valid ELF, then invoke \p Callback on the ELFObjectFileBase
/// created from this range, otherwise, return 0.
/// If \p Callback is invoked, then return whatever value \p Callback returns.
template <typename F>
static int32_t withBytesAsElf(char *BytesBegin, char *BytesEnd, F Callback) {
  size_t Size = BytesEnd - BytesBegin;
  StringRef StrBuf(BytesBegin, Size);

  auto Magic = identify_magic(StrBuf);
  if (Magic != file_magic::elf && Magic != file_magic::elf_relocatable &&
      Magic != file_magic::elf_executable &&
      Magic != file_magic::elf_shared_object && Magic != file_magic::elf_core) {
    DP("Not an ELF image!\n");
    return 0;
  }

  std::unique_ptr<MemoryBuffer> MemBuf =
      MemoryBuffer::getMemBuffer(StrBuf, "", false);
  Expected<std::unique_ptr<ObjectFile>> BinOrErr =
      ObjectFile::createELFObjectFile(MemBuf->getMemBufferRef(),
                                      /*InitContent=*/false);
  if (!BinOrErr) {
    DP("Unable to get ELF handle: %s!\n",
       toString(BinOrErr.takeError()).c_str());
    return 0;
  }

  auto *Object = dyn_cast<const ELFObjectFileBase>(BinOrErr->get());

  if (!Object) {
    DP("Unknown ELF format!\n");
    return 0;
  }

  return Callback(Object);
}

// Check whether an image is valid for execution on target_id
int32_t elf_check_machine(__tgt_device_image *image, uint16_t target_id) {
  auto CheckMachine = [target_id](const ELFObjectFileBase *Object) {
    return target_id == Object->getEMachine();
  };
  return withBytesAsElf(reinterpret_cast<char *>(image->ImageStart),
                        reinterpret_cast<char *>(image->ImageEnd),
                        CheckMachine);
}

int32_t elf_is_dynamic(__tgt_device_image *image) {
  auto CheckDynType = [](const ELFObjectFileBase *Object) {
    uint16_t Type = Object->getEType();
    DP("ELF Type: %d\n", Type);
    return Type == ET_DYN;
  };
  return withBytesAsElf(reinterpret_cast<char *>(image->ImageStart),
                        reinterpret_cast<char *>(image->ImageEnd),
                        CheckDynType);
}
