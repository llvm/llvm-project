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
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>

namespace llvm {
namespace dsymutil {

Error PseudoProbeLinker::collect(const object::ObjectFile &Obj) {
  const auto *MO = dyn_cast<object::MachOObjectFile>(&Obj);
  if (!MO)
    return Error::success();

  for (const object::SectionRef &Section : MO->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();

    std::string *Dest = StringSwitch<std::string *>(*NameOrErr)
                            .Case("__probes", &Probes)
                            .Case("__probe_descs", &ProbeDescs)
                            .Default(nullptr);
    if (!Dest)
      continue;

    assert(Section.relocations().empty() &&
           "pseudo-probe section unexpectedly carries relocations");
    if (!Section.relocations().empty())
      return createStringError(
          inconvertibleErrorCode(),
          "unexpected relocations in pseudo-probe section " + *NameOrErr);

    Expected<StringRef> ContentsOrErr = Section.getContents();
    if (!ContentsOrErr)
      return ContentsOrErr.takeError();
    *Dest += *ContentsOrErr;
  }
  return Error::success();
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

  // Write pseudo_probes metadata.
  Path.clear();
  sys::path::append(Path, *Options.ResourceDir, "Profiling",
                    "pseudo_probes" + Suffix);
  std::error_code EC;
  raw_fd_ostream ProbesOS(Path.str(), EC, sys::fs::OF_None);
  if (EC)
    return errorCodeToError(EC);
  ProbesOS << getProbes();

  // Write pseudo_probe_descs metadata.
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
