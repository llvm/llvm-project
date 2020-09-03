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
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

void jitLink_ELF(std::unique_ptr<JITLinkContext> Ctx) {

  // We don't want to do full ELF validation here. We just verify it is elf'ish.
  // Probably should parse into an elf header when we support more than x86 :)

  StringRef Data = Ctx->getObjectBuffer().getBuffer();
  if (Data.size() < llvm::ELF::EI_MAG3 + 1) {
    Ctx->notifyFailed(make_error<JITLinkError>("Truncated ELF buffer"));
    return;
  }

  if (!memcmp(Data.data(), llvm::ELF::ElfMagic, strlen(llvm::ELF::ElfMagic))) {
    if (Data.data()[llvm::ELF::EI_CLASS] == ELF::ELFCLASS64) {
      return jitLink_ELF_x86_64(std::move(Ctx));
    }
  }

  Ctx->notifyFailed(make_error<JITLinkError>("ELF magic not valid"));
}

} // end namespace jitlink
} // end namespace llvm
