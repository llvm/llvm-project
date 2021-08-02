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
#include "llvm/ExecutionEngine/JITLink/ELF_riscv.h"
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

Expected<std::unique_ptr<LinkGraph>>
createLinkGraphFromELFObject(MemoryBufferRef ObjectBuffer) {
  StringRef Buffer = ObjectBuffer.getBuffer();
  if (Buffer.size() < ELF::EI_MAG3 + 1)
    return make_error<JITLinkError>("Truncated ELF buffer");

  if (memcmp(Buffer.data(), ELF::ElfMagic, strlen(ELF::ElfMagic)) != 0)
    return make_error<JITLinkError>("ELF magic not valid");

  Expected<uint16_t> TargetMachineArch = readTargetMachineArch(Buffer);
  if (!TargetMachineArch)
    return TargetMachineArch.takeError();

  switch (*TargetMachineArch) {
  case ELF::EM_RISCV:
    return createLinkGraphFromELFObject_riscv(ObjectBuffer);
  case ELF::EM_X86_64:
    return createLinkGraphFromELFObject_x86_64(ObjectBuffer);
  default:
    return make_error<JITLinkError>(
        "Unsupported target machine architecture in ELF object " +
        ObjectBuffer.getBufferIdentifier());
  }
}

void link_ELF(std::unique_ptr<LinkGraph> G,
              std::unique_ptr<JITLinkContext> Ctx) {
  switch (G->getTargetTriple().getArch()) {
  case Triple::riscv32:
  case Triple::riscv64:
    link_ELF_riscv(std::move(G), std::move(Ctx));
    return;
  case Triple::x86_64:
    link_ELF_x86_64(std::move(G), std::move(Ctx));
    return;
  default:
    Ctx->notifyFailed(make_error<JITLinkError>(
        "Unsupported target machine architecture in ELF link graph " +
        G->getName()));
    return;
  }
}

} // end namespace jitlink
} // end namespace llvm
