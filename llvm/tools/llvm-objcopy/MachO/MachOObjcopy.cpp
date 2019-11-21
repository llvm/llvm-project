//===- MachOObjcopy.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MachOObjcopy.h"
#include "../CopyConfig.h"
#include "MachOReader.h"
#include "MachOWriter.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace objcopy {
namespace macho {

using namespace object;
using SectionPred = std::function<bool(const Section &Sec)>;

static void removeSections(const CopyConfig &Config, Object &Obj) {
  SectionPred RemovePred = [](const Section &) { return false; };

  if (!Config.ToRemove.empty()) {
    RemovePred = [&Config, RemovePred](const Section &Sec) {
      return Config.ToRemove.matches(Sec.CanonicalName);
    };
  }

  if (Config.StripAll || Config.StripDebug) {
    // Remove all debug sections.
    RemovePred = [RemovePred](const Section &Sec) {
      if (Sec.Segname == "__DWARF")
        return true;

      return RemovePred(Sec);
    };
  }

  if (!Config.OnlySection.empty()) {
    // Overwrite RemovePred because --only-section takes priority.
    RemovePred = [&Config](const Section &Sec) {
      return !Config.OnlySection.matches(Sec.CanonicalName);
    };
  }

  return Obj.removeSections(RemovePred);
}

static void markSymbols(const CopyConfig &Config, Object &Obj) {
  // Symbols referenced from the indirect symbol table must not be removed.
  for (IndirectSymbolEntry &ISE : Obj.IndirectSymTable.Symbols)
    if (ISE.Symbol)
      (*ISE.Symbol)->Referenced = true;
}

static void updateAndRemoveSymbols(const CopyConfig &Config, Object &Obj) {
  for (SymbolEntry &Sym : Obj.SymTable) {
    auto I = Config.SymbolsToRename.find(Sym.Name);
    if (I != Config.SymbolsToRename.end())
      Sym.Name = I->getValue();
  }

  auto RemovePred = [Config](const std::unique_ptr<SymbolEntry> &N) {
    if (N->Referenced)
      return false;
    return Config.StripAll;
  };

  Obj.SymTable.removeSymbols(RemovePred);
}

static LoadCommand buildRPathLoadCommand(StringRef Path) {
  LoadCommand LC;
  MachO::rpath_command RPathLC;
  RPathLC.cmd = MachO::LC_RPATH;
  RPathLC.path = sizeof(MachO::rpath_command);
  RPathLC.cmdsize = alignTo(sizeof(MachO::rpath_command) + Path.size(), 8);
  LC.MachOLoadCommand.rpath_command_data = RPathLC;
  LC.Payload.assign(RPathLC.cmdsize - sizeof(MachO::rpath_command), 0);
  std::copy(Path.begin(), Path.end(), LC.Payload.begin());
  return LC;
}

static Error handleArgs(const CopyConfig &Config, Object &Obj) {
  if (Config.AllowBrokenLinks || !Config.BuildIdLinkDir.empty() ||
      Config.BuildIdLinkInput || Config.BuildIdLinkOutput ||
      !Config.SplitDWO.empty() || !Config.SymbolsPrefix.empty() ||
      !Config.AllocSectionsPrefix.empty() || !Config.AddSection.empty() ||
      !Config.DumpSection.empty() || !Config.KeepSection.empty() ||
      Config.NewSymbolVisibility || !Config.SymbolsToGlobalize.empty() ||
      !Config.SymbolsToKeep.empty() || !Config.SymbolsToLocalize.empty() ||
      !Config.SymbolsToWeaken.empty() || !Config.SymbolsToKeepGlobal.empty() ||
      !Config.SectionsToRename.empty() ||
      !Config.UnneededSymbolsToRemove.empty() ||
      !Config.SetSectionAlignment.empty() || !Config.SetSectionFlags.empty() ||
      Config.ExtractDWO || Config.KeepFileSymbols || Config.LocalizeHidden ||
      Config.PreserveDates || Config.StripAllGNU || Config.StripDWO ||
      Config.StripNonAlloc || Config.StripSections || Config.Weaken ||
      Config.DecompressDebugSections || Config.StripNonAlloc ||
      Config.StripSections || Config.StripUnneeded ||
      Config.DiscardMode != DiscardType::None || !Config.SymbolsToAdd.empty() ||
      Config.EntryExpr) {
    return createStringError(llvm::errc::invalid_argument,
                             "option not supported by llvm-objcopy for MachO");
  }
  removeSections(Config, Obj);

  // Mark symbols to determine which symbols are still needed.
  if (Config.StripAll)
    markSymbols(Config, Obj);

  updateAndRemoveSymbols(Config, Obj);

  if (Config.StripAll)
    for (LoadCommand &LC : Obj.LoadCommands)
      for (Section &Sec : LC.Sections)
        Sec.Relocations.clear();

  for (StringRef RPath : Config.RPathToAdd) {
    for (LoadCommand &LC : Obj.LoadCommands) {
      if (LC.MachOLoadCommand.load_command_data.cmd == MachO::LC_RPATH &&
          RPath == StringRef(reinterpret_cast<char *>(LC.Payload.data()),
                             LC.Payload.size())
                       .trim(0)) {
        return createStringError(errc::invalid_argument,
                                 "rpath " + RPath +
                                     " would create a duplicate load command");
      }
    }
    Obj.addLoadCommand(buildRPathLoadCommand(RPath));
  }
  return Error::success();
}

Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::MachOObjectFile &In, Buffer &Out) {
  MachOReader Reader(In);
  std::unique_ptr<Object> O = Reader.create();
  if (!O)
    return createFileError(
        Config.InputFilename,
        createStringError(object_error::parse_failed,
                          "unable to deserialize MachO object"));

  if (Error E = handleArgs(Config, *O))
    return createFileError(Config.InputFilename, std::move(E));

  // TODO: Support 16KB pages which are employed in iOS arm64 binaries:
  //       https://github.com/llvm/llvm-project/commit/1bebb2832ee312d3b0316dacff457a7a29435edb
  const uint64_t PageSize = 4096;

  MachOWriter Writer(*O, In.is64Bit(), In.isLittleEndian(), PageSize, Out);
  if (auto E = Writer.finalize())
    return E;
  return Writer.write();
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
