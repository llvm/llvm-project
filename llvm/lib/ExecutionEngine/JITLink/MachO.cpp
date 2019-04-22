//===-------------- MachO.cpp - JIT linker function for MachO -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachO jit-link function.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/JITLink/MachO.h"

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/ExecutionEngine/JITLink/MachO_x86_64.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

#define DEBUG_TYPE "jitlink"

namespace llvm {
namespace jitlink {

void jitLink_MachO(std::unique_ptr<JITLinkContext> Ctx) {

  // We don't want to do full MachO validation here. Just parse enough of the
  // header to find out what MachO linker to use.

  StringRef Data = Ctx->getObjectBuffer().getBuffer();
  if (Data.size() < 4) {
    Ctx->notifyFailed(make_error<JITLinkError>("Truncated MachO buffer"));
    return;
  }

  uint32_t Magic;
  memcpy(&Magic, Data.data(), sizeof(uint32_t));
  LLVM_DEBUG({
    dbgs() << "jitLink_MachO: magic = " << format("0x%08" PRIx32, Magic)
           << "\n";
  });

  if (Magic == MachO::MH_MAGIC || Magic == MachO::MH_CIGAM) {
    Ctx->notifyFailed(
        make_error<JITLinkError>("MachO 32-bit platforms not supported"));
    return;
  } else if (Magic == MachO::MH_MAGIC_64 || Magic == MachO::MH_CIGAM_64) {
    MachO::mach_header_64 Header;

    memcpy(&Header, Data.data(), sizeof(MachO::mach_header_64));
    if (Magic == MachO::MH_CIGAM_64)
      swapStruct(Header);

    LLVM_DEBUG({
      dbgs() << "jitLink_MachO: cputype = "
             << format("0x%08" PRIx32, Header.cputype)
             << ", cpusubtype = " << format("0x%08" PRIx32, Header.cpusubtype)
             << "\n";
    });

    switch (Header.cputype) {
    case MachO::CPU_TYPE_X86_64:
      return jitLink_MachO_x86_64(std::move(Ctx));
    }
    Ctx->notifyFailed(make_error<JITLinkError>("MachO-64 CPU type not valid"));
    return;
  }

  Ctx->notifyFailed(make_error<JITLinkError>("MachO magic not valid"));
}

} // end namespace jitlink
} // end namespace llvm
