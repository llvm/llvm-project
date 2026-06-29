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

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/ObjectFile.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>

using namespace llvm;
using namespace omp;
using namespace target;

namespace {

/// We calculate the program counter relative to the RPC client symbol.
constexpr StringRef RPCClientSymbol = "__llvm_rpc_client";

/// Link-time ELF value of \p Name in \p Obj, if the image defines it.
std::optional<uint64_t> getSymbolValue(const object::ObjectFile &Obj,
                                       StringRef Name) {
  Expected<std::optional<object::ELFSymbolRef>> SymOrErr =
      utils::elf::getSymbol(Obj, Name);
  if (!SymOrErr) {
    consumeError(SymOrErr.takeError());
    return std::nullopt;
  }
  if (!*SymOrErr)
    return std::nullopt;
  Expected<uint64_t> ValueOrErr = (*SymOrErr)->getValue();
  if (!ValueOrErr) {
    consumeError(ValueOrErr.takeError());
    return std::nullopt;
  }
  return *ValueOrErr;
}

/// A loaded image selected to symbolize a report against, holding the data
/// needed to rebase raw device addresses into image VAs.
struct ResolvedImage {
  std::unique_ptr<object::ObjectFile> Obj;
  std::unique_ptr<DWARFContext> DICtx;
  uint64_t ClientValue = 0;
  uint64_t ClientDeviceAddr = 0;

  explicit operator bool() const { return static_cast<bool>(Obj); }

  /// Map a raw device address to its address inside the resolved image.
  uint64_t rebase(uint64_t DeviceAddr) const {
    return DeviceAddr - ClientDeviceAddr + ClientValue;
  }
};

/// Find the loaded image that contains this program counter, using the RPC
/// client symbol as a reference point. The relative offset of the symbol from
/// the device and the original ELF image lets us identify the program counter.
ResolvedImage selectImage(plugin::GenericDeviceTy &Device, uint64_t PC) {
  plugin::GenericGlobalHandlerTy &Handler = Device.Plugin.getGlobalHandler();
  ResolvedImage Result;
  for (plugin::DeviceImageTy *Image : Device.LoadedImages) {
    if (!Image)
      continue;
    Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
        object::ObjectFile::createELFObjectFile(Image->getMemoryBuffer());
    if (!ObjOrErr) {
      consumeError(ObjOrErr.takeError());
      continue;
    }
    std::optional<uint64_t> CV = getSymbolValue(**ObjOrErr, RPCClientSymbol);
    if (!CV)
      continue;

    plugin::GlobalTy ClientGlobal(RPCClientSymbol.str());
    if (auto Err =
            Handler.getGlobalMetadataFromDevice(Device, *Image, ClientGlobal)) {
      consumeError(std::move(Err));
      continue;
    }
    uint64_t Base = reinterpret_cast<uintptr_t>(ClientGlobal.getPtr());
    uint64_t ImagePC = PC - Base + *CV;
    // '__builtin_return_address' yields the return PC, step back into the call.
    bool PCResolved = !utils::elf::findFunctionSymbol(
                           **ObjOrErr, ImagePC ? ImagePC - 1 : ImagePC)
                           .empty();
    if (!Result.Obj || PCResolved) {
      Result.ClientValue = *CV;
      Result.ClientDeviceAddr = Base;
      Result.DICtx = DWARFContext::create(**ObjOrErr);
      Result.Obj = std::move(*ObjOrErr);
    }
    if (PCResolved)
      break;
  }
  return Result;
}

} // namespace

// Deduplicates reports on the same program counter and kind, so a whole grid
// tripping one check yields a single diagnostic.
bool llvm::omp::target::SanitizerTables::isNewReport(uint64_t PC,
                                                     StringRef Kind) {
  SmallString<64> Key(Kind);
  Key += ':' + utohexstr(PC);
  std::lock_guard<std::mutex> Guard(Mtx);
  return Reports.insert(Key).second;
}

void llvm::omp::target::reportGPUUBSan(plugin::GenericDeviceTy &Device,
                                       SanitizerTables &Tables,
                                       const __ubsan_gpu_report &R) {
  // The device sends a fixed-width, null-terminated kind.
  size_t KindLen = strnlen(R.kind, UBSAN_GPU_KIND_MAX);
  StringRef Kind(R.kind, KindLen);
  if (Kind.empty())
    return;

  // Drop duplicates before the expensive ELF/DWARF parsing below.
  if (!Tables.isNewReport(R.pc, Kind))
    return;

  ResolvedImage Image = selectImage(Device, R.pc);

  uint64_t ImagePC = Image ? Image.rebase(R.pc) : R.pc;
  uint64_t LookupPC = ImagePC ? ImagePC - 1 : ImagePC;

  SmallVector<utils::elf::SourceLocation> Frames;
  if (Image)
    Frames = utils::elf::symbolize(*Image.DICtx, LookupPC);

  // Without debug info there is nothing to symbolize, so fall back to the same
  // minimal line the device runtime prints on the host.
  if (Frames.empty()) {
    fprintf(stderr, "ubsan: %s by 0x%" PRIx64 "\n", Kind.str().c_str(),
            ImagePC);
    return;
  }

  const utils::elf::SourceLocation &Top = Frames.front();
  StringRef TopFunc = Top.FunctionName;
  std::string Loc;
  if (!Top.FileName.empty())
    Loc =
        (Top.FileName + ":" + Twine(Top.Line) + ":" + Twine(Top.Column)).str();

  if (!Loc.empty())
    fprintf(stderr, "%s: runtime error: %s\n", Loc.c_str(), Kind.str().c_str());
  else
    fprintf(stderr, "runtime error: %s\n", Kind.str().c_str());
  fprintf(stderr,
          "    on GPU thread: block (%u,%u,%u) thread (%u,%u,%u) lane %u\n",
          R.block[0], R.block[1], R.block[2], R.thread[0], R.thread[1],
          R.thread[2], R.lane);

  for (uint32_t I = 0, N = static_cast<uint32_t>(Frames.size()); I < N; ++I) {
    const utils::elf::SourceLocation &F = Frames[I];
    fprintf(stderr, "    #%u 0x%" PRIx64 " in %s %s:%u:%u\n", I, ImagePC,
            F.FunctionName.empty() ? "??" : F.FunctionName.c_str(),
            F.FileName.empty() ? "??" : F.FileName.c_str(), F.Line, F.Column);
  }

  fprintf(stderr, "SUMMARY: UndefinedBehaviorSanitizer: %s%s%s%s%s\n",
          Kind.str().c_str(), Loc.empty() ? "" : " ", Loc.c_str(),
          TopFunc.empty() ? "" : " in ",
          TopFunc.empty() ? "" : TopFunc.str().c_str());
}
