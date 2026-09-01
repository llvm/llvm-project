//===-- Sanitizer.cpp - Host-side GPU sanitizer reporting -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Sanitizer.h"

#include "PluginInterface.h"
#include "Utils/ELF.h"

#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <utility>

using namespace llvm;
using namespace omp;
using namespace target;

namespace {

// The sentinel image we use to identify the proper PC location.
constexpr StringRef RPCClientSymbol = "__llvm_rpc_client";

// Link-time ELF value of \p Name in \p Obj, if the image defines it.
std::optional<uint64_t> getSymbolValue(const object::ObjectFile &Obj,
                                       StringRef Name) {
  Expected<std::optional<object::ELFSymbolRef>> SymOrErr =
      utils::elf::getSymbol(Obj, Name);
  if (!SymOrErr)
    return consumeError(SymOrErr.takeError()), std::nullopt;
  if (!*SymOrErr)
    return std::nullopt;
  Expected<uint64_t> ValueOrErr = (*SymOrErr)->getValue();
  if (!ValueOrErr)
    return consumeError(ValueOrErr.takeError()), std::nullopt;
  return *ValueOrErr;
}

// A loaded image selected to symbolize a report against, holding the data
// needed to rebase raw device addresses into image VAs.
struct ResolvedImage {
  std::unique_ptr<object::ObjectFile> Obj;
  std::unique_ptr<DWARFContext> DICtx;
  int64_t Bias = 0;

  explicit operator bool() const { return Obj != nullptr; }
  uintptr_t getAddress(uintptr_t DeviceAddr) const { return DeviceAddr + Bias; }
};

// Returns the difference between the RPC client symbol on the device and in the
// image to get the relative offset.
std::optional<int64_t> getImageOffset(plugin::GenericDeviceTy &Device,
                                      plugin::DeviceImageTy &Image,
                                      const object::ObjectFile &Obj) {
  std::optional<uintptr_t> ImageVA = getSymbolValue(Obj, RPCClientSymbol);
  if (!ImageVA)
    return std::nullopt;
  plugin::GlobalTy Client(RPCClientSymbol.str());
  if (Error Err = Device.Plugin.getGlobalHandler().getGlobalMetadataFromDevice(
          Device, Image, Client))
    return consumeError(std::move(Err)), std::nullopt;
  return static_cast<int64_t>(*ImageVA) -
         static_cast<int64_t>(reinterpret_cast<uintptr_t>(Client.getPtr()));
}

// Identify which of the loaded images contains our program-counter. We use the
// address of the RPC client as our sentinel to find the relative offset.
ResolvedImage resolve(plugin::GenericDeviceTy &Device, uintptr_t PC) {
  for (plugin::DeviceImageTy *Image : Device.LoadedImages) {
    if (!Image)
      continue;

    Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
        object::ObjectFile::createObjectFile(Image->getMemoryBuffer());
    if (!ObjOrErr) {
      consumeError(ObjOrErr.takeError());
      continue;
    }

    std::optional<int64_t> Bias = getImageOffset(Device, *Image, **ObjOrErr);
    if (!Bias)
      continue;

    // Step back into the call site from the saved return address.
    uintptr_t LookupPC = PC ? PC - 1 : PC;
    if (utils::elf::findFunctionSymbol(**ObjOrErr, LookupPC + *Bias).empty())
      continue;

    ResolvedImage RI;
    RI.Obj = std::move(*ObjOrErr);
    RI.DICtx = DWARFContext::create(*RI.Obj);
    RI.Bias = *Bias;
    return RI;
  }
  return {};
}

// Print the source frames for a device PC, one line per inlined frame.
void printBacktrace(raw_ostream &OS, const ResolvedImage &RI, uintptr_t PC) {
  uintptr_t ImagePC = RI.getAddress(PC);
  // The device records return addresses; step back into the call site.
  uintptr_t LookupPC = ImagePC ? ImagePC - 1 : ImagePC;
  SmallVector<utils::elf::SourceLocation> Frames =
      RI ? utils::elf::symbolize(*RI.DICtx, LookupPC)
         : SmallVector<utils::elf::SourceLocation>();
  if (Frames.empty()) {
    StringRef Fn =
        RI ? utils::elf::findFunctionSymbol(*RI.Obj, LookupPC) : StringRef();
    OS << formatv("==CSAN==     #0 {0} ({1:x})\n",
                  Fn.empty() ? "??" : demangle(Fn), ImagePC);
    return;
  }
  for (auto [I, Frame] : enumerate(Frames))
    OS << formatv("==CSAN==     #{0} {1} {2}:{3}:{4} ({5:x})\n", I,
                  Frame.FunctionName.empty() ? "??" : Frame.FunctionName,
                  Frame.FileName.empty() ? "??" : Frame.FileName, Frame.Line,
                  Frame.Column, ImagePC);
}

void reportRace(raw_ostream &OS, const ResolvedImage &RI,
                const __tsan_gpu_race &Race) {
  StringRef Op = (Race.access_type & TSAN_GPU_ACCESS_COMPOUND)
                     ? "Read-modify-write"
                 : (Race.access_type & TSAN_GPU_ACCESS_WRITE) ? "Write"
                                                              : "Read";
  StringRef Atomic =
      (Race.access_type & TSAN_GPU_ACCESS_ATOMIC) ? "atomic " : "";
  StringRef Kind = Race.kind == TSAN_GPU_UNKNOWN_ORIGIN
                       ? "data race (unknown origin)"
                   : Race.kind == TSAN_GPU_INTRA_WAVE ? "data race (intra-wave)"
                                                      : "data race";

  OS << formatv("==CSAN== WARNING: ConcurrencySanitizer: {0}\n", Kind);
  OS << formatv("==CSAN==   {0}{1} of size {2} at {3:x} in block "
                "({4},{5},{6}) thread ({7},{8},{9}) lane {10}\n",
                Atomic, Op, Race.size, Race.addr, Race.block[0], Race.block[1],
                Race.block[2], Race.thread[0], Race.thread[1], Race.thread[2],
                Race.lane);

  printBacktrace(OS, RI, Race.pc);
  if (Race.kind == TSAN_GPU_INTRA_WAVE) {
    // The conflicting lane executed the same instruction, so it shares the PC.
    OS << formatv("==CSAN==   Previous access in block ({0},{1},{2}) thread "
                  "({3},{4},{5}) lane {6}\n",
                  Race.block[0], Race.block[1], Race.block[2],
                  Race.peer_thread[0], Race.peer_thread[1], Race.peer_thread[2],
                  Race.peer_lane);
    printBacktrace(OS, RI, Race.pc);
  } else if (Race.peer_pc) {
    OS << "==CSAN==   Previous access:\n";
    printBacktrace(OS, RI, Race.peer_pc);
  }
  if (RI) {
    StringRef Var =
        utils::elf::findDataSymbol(*RI.Obj, RI.getAddress(Race.addr));
    if (!Var.empty())
      OS << formatv("==CSAN==   Address {0:x} is global variable '{1}'\n",
                    Race.addr, Var);
  }
}

} // namespace

// Deduplicates races on the conflicting program counters, similar to TSan.
bool llvm::omp::target::SanitizerTables::isNewRace(uintptr_t PC,
                                                   uintptr_t PeerPC,
                                                   unsigned Kind) {
  if (PC > PeerPC)
    std::swap(PC, PeerPC);
  std::lock_guard<std::mutex> Guard(Mtx);
  return Races.insert({PC ^ (static_cast<uint64_t>(Kind) << 56), PeerPC})
      .second;
}

void llvm::omp::target::reportGPUCSanRace(plugin::GenericDeviceTy &Device,
                                          SanitizerTables &Tables,
                                          const __tsan_gpu_race &Race) {
  // Drop duplicates before the expensive ELF/DWARF parsing below.
  if (!Tables.isNewRace(Race.pc, Race.peer_pc, Race.kind))
    return;

  reportRace(errs(), resolve(Device, Race.pc), Race);
}
