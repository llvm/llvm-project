//===- tools/dsymutil/PseudoProbeLinker.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PseudoProbeLinker.h"
#include "BinaryHolder.h"
#include "DebugMap.h"
#include "LinkUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace dsymutil {

Error PseudoProbeLinker::emit(const Triple &TheTriple) const {
  // Don't create the directory and the file if there is nothing to write.
  if (empty())
    return Error::success();

  // Without a resource directory (e.g. --flat) there is no dSYM bundle to place
  // the content. Nothing to do in this case.
  if (!Options.ResourceDir)
    return Error::success();

  SmallString<128> Path;
  sys::path::append(Path, *Options.ResourceDir, "Profiling");
  if (std::error_code EC = sys::fs::create_directories(Path.str(), true,
                                                       sys::fs::perms::all_all))
    return errorCodeToError(EC);

  // For fat binaries, also append a dash and the architecture name.
  sys::path::append(Path, "pseudo_probes");
  if (Options.NumDebugMaps > 1) {
    Path += '-';
    Path += TheTriple.getArchName();
  }

  // Build the minimal MC machinery needed to write a MachO object holding the
  // probe sections. All targets and their MC layers are registered by
  // dsymutil_main.
  std::string TripleName = TheTriple.getTriple();
  std::string ErrorStr;
  const Target *TheTarget = TargetRegistry::lookupTarget(TheTriple, ErrorStr);
  if (!TheTarget)
    return createStringError(inconvertibleErrorCode(), ErrorStr);

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TheTriple));
  if (!MRI)
    return createStringError(inconvertibleErrorCode(),
                             "no register info for target " + TripleName);

  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TheTriple, MCOptions));
  if (!MAI)
    return createStringError(inconvertibleErrorCode(),
                             "no asm info for target " + TripleName);

  std::unique_ptr<MCSubtargetInfo> MSTI(
      TheTarget->createMCSubtargetInfo(TheTriple, "", ""));
  if (!MSTI)
    return createStringError(inconvertibleErrorCode(),
                             "no subtarget info for target " + TripleName);

  MCContext Ctx(TheTriple, *MAI, *MRI, *MSTI);
  std::unique_ptr<MCObjectFileInfo> MOFI(
      TheTarget->createMCObjectFileInfo(Ctx, /*PIC=*/false));
  Ctx.setObjectFileInfo(MOFI.get());

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCCodeEmitter> MCE(TheTarget->createMCCodeEmitter(*MII, Ctx));
  std::unique_ptr<MCAsmBackend> MAB(
      TheTarget->createMCAsmBackend(*MSTI, *MRI, MCOptions));
  if (!MCE || !MAB)
    return createStringError(inconvertibleErrorCode(),
                             "no code emitter/asm backend for target " +
                                 TripleName);

  std::error_code EC;
  raw_fd_ostream OS(Options.NoOutput ? "-" : Path.str(), EC, sys::fs::OF_None);
  if (EC)
    return errorCodeToError(EC);

  std::unique_ptr<MCObjectWriter> OW = MAB->createObjectWriter(OS);
  std::unique_ptr<MCStreamer> MS(TheTarget->createMCObjectStreamer(
      TheTriple, Ctx, std::move(MAB), std::move(OW), std::move(MCE), *MSTI));
  if (!MS)
    return createStringError(inconvertibleErrorCode(),
                             "no object streamer for target " + TripleName);

  auto emitSection = [&](StringRef Name, StringRef Data) {
    if (Data.empty())
      return;
    MCSectionMachO *Sec = Ctx.getMachOSection(
        "__LLVM", Name, /*TypeAndAttributes=*/0, SectionKind::getMetadata());
    MS->switchSection(Sec);
    MS->emitBytes(Data);
  };
  emitSection("__probes", getProbes());
  emitSection("__probe_descs", getProbeDescs());
  MS->finish();

  return Error::success();
}

bool PseudoProbeLinker::link(const DebugMap &Map) {
  for (const auto &Obj : Map.objects()) {
    // Load the object. Any load error is reported by the DWARF link; skip
    // silently here to avoid duplicate diagnostics.
    auto ObjectEntry =
        BinHolder.getObjectEntry(Obj->getObjectFilename(), Obj->getTimestamp());
    if (!ObjectEntry) {
      consumeError(ObjectEntry.takeError());
      continue;
    }
    auto Object = ObjectEntry->getObject(Map.getTriple());
    if (!Object) {
      consumeError(Object.takeError());
      continue;
    }

    collect(*Object);
  }

  if (Error E = emit(Map.getTriple())) {
    handleAllErrors(std::move(E), [](const ErrorInfoBase &EI) {
      WithColor::error() << EI.message() << '\n';
    });
    return false;
  }
  return true;
}

void PseudoProbeLinker::collect(const object::ObjectFile &Obj) {
  const auto *MO = dyn_cast<object::MachOObjectFile>(&Obj);
  if (!MO)
    return;

  for (const object::SectionRef &Section : MO->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }
    std::string *Dest = nullptr;
    if (*NameOrErr == "__probes")
      Dest = &Probes;
    else if (*NameOrErr == "__probe_descs")
      Dest = &ProbeDescs;
    else
      continue;

    Expected<StringRef> ContentsOrErr = Section.getContents();
    if (!ContentsOrErr) {
      consumeError(ContentsOrErr.takeError());
      continue;
    }
    *Dest += *ContentsOrErr;
  }
}
} // end namespace dsymutil
} // end namespace llvm
