//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PseudoProbeLinker.h"
#include "LinkUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace dsymutil {

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

    if (!Section.relocations().empty())
      report_fatal_error(
          Twine("unexpected relocations in pseudo-probe section ") + *NameOrErr,
          /*GenCrashDiag=*/false);

    Expected<StringRef> ContentsOrErr = Section.getContents();
    if (!ContentsOrErr) {
      consumeError(ContentsOrErr.takeError());
      continue;
    }
    *Dest += *ContentsOrErr;
  }
}

Error PseudoProbeLinker::emit(const Triple &TheTriple) const {
  if (empty())
    return Error::success();

  if (!Options.ResourceDir || Options.NoOutput)
    return Error::success();

  SmallString<128> Path;
  sys::path::append(Path, *Options.ResourceDir, "Profiling");
  if (std::error_code EC = sys::fs::create_directories(Path.str(), true,
                                                       sys::fs::perms::all_all))
    return errorCodeToError(EC);

  std::string Suffix;
  if (Options.NumDebugMaps > 1)
    Suffix = ("-" + TheTriple.getArchName()).str();

  // write pseudo_probes metadata
  Path.clear();
  sys::path::append(Path, *Options.ResourceDir, "Profiling",
                    "pseudo_probes" + Suffix);
  std::error_code EC;
  raw_fd_ostream ProbesOS(Path.str(), EC, sys::fs::OF_None);
  if (EC)
    return errorCodeToError(EC);
  ProbesOS << getProbes();

  // write pseudo_probe_descs metadata
  Path.clear();
  sys::path::append(Path, *Options.ResourceDir, "Profiling",
                    "pseudo_probe_descs" + Suffix);
  raw_fd_ostream DescsOS(Path.str(), EC, sys::fs::OF_None);
  if (EC)
    return errorCodeToError(EC);
  DescsOS << getProbeDescs();

  return Error::success();
}

} // end namespace dsymutil
} // end namespace llvm
