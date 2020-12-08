//===-------------- ELF.cpp - JIT linker function for ELF -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ELF jit-link function.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/ELF.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/ELF_x86_64.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

Expected<uint16_t> readTargetMachineArch(StringRef Buffer) {
  const char *Data = Buffer.data();

  if (Data[ELF::EI_DATA] == ELF::ELFDATA2LSB) {
    if (Data[ELF::EI_CLASS] == ELF::ELFCLASS64) {
      if (auto File = llvm::object::ELF64LEFile::create(Buffer)) {
        return File->getHeader().e_machine;
      } else {
        return File.takeError();
      }
    } else if (Data[ELF::EI_CLASS] == ELF::ELFCLASS32) {
      if (auto File = llvm::object::ELF32LEFile::create(Buffer)) {
        return File->getHeader().e_machine;
      } else {
        return File.takeError();
      }
    }
  }

  return ELF::EM_NONE;
}

void jitLink_ELF(std::unique_ptr<JITLinkContext> Ctx) {
  StringRef Buffer = Ctx->getObjectBuffer().getBuffer();
  if (Buffer.size() < ELF::EI_MAG3 + 1) {
    Ctx->notifyFailed(make_error<JITLinkError>("Truncated ELF buffer"));
    return;
  }

  if (memcmp(Buffer.data(), ELF::ElfMagic, strlen(ELF::ElfMagic)) != 0) {
    Ctx->notifyFailed(make_error<JITLinkError>("ELF magic not valid"));
    return;
  }

  Expected<uint16_t> TargetMachineArch = readTargetMachineArch(Buffer);
  if (!TargetMachineArch) {
    Ctx->notifyFailed(TargetMachineArch.takeError());
    return;
  }

  switch (*TargetMachineArch) {
  case ELF::EM_X86_64:
    jitLink_ELF_x86_64(std::move(Ctx));
    return;
  default:
    Ctx->notifyFailed(make_error<JITLinkError>(
        "Unsupported target machine architecture in ELF object " +
        Ctx->getObjectBuffer().getBufferIdentifier()));
    return;
  }
}

} // end namespace jitlink
} // end namespace llvm
