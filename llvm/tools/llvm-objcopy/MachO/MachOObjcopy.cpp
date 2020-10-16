//===- MachOObjcopy.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MachOObjcopy.h"
#include "../CopyConfig.h"
#include "../llvm-objcopy.h"
#include "MachOReader.h"
#include "MachOWriter.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/MachOUniversalWriter.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace objcopy {
namespace macho {

using namespace object;
using SectionPred = std::function<bool(const std::unique_ptr<Section> &Sec)>;
using LoadCommandPred = std::function<bool(const LoadCommand &LC)>;

#ifndef NDEBUG
static bool isLoadCommandWithPayloadString(const LoadCommand &LC) {
  // TODO: Add support for LC_REEXPORT_DYLIB, LC_LOAD_UPWARD_DYLIB and
  // LC_LAZY_LOAD_DYLIB
  return LC.MachOLoadCommand.load_command_data.cmd == MachO::LC_RPATH ||
         LC.MachOLoadCommand.load_command_data.cmd == MachO::LC_ID_DYLIB ||
         LC.MachOLoadCommand.load_command_data.cmd == MachO::LC_LOAD_DYLIB ||
         LC.MachOLoadCommand.load_command_data.cmd == MachO::LC_LOAD_WEAK_DYLIB;
}
#endif

static StringRef getPayloadString(const LoadCommand &LC) {
  assert(isLoadCommandWithPayloadString(LC) &&
         "unsupported load command encountered");

  return StringRef(reinterpret_cast<const char *>(LC.Payload.data()),
                   LC.Payload.size())
      .rtrim('\0');
}

static Error removeSections(const CopyConfig &Config, Object &Obj) {
  SectionPred RemovePred = [](const std::unique_ptr<Section> &) {
    return false;
  };

  if (!Config.ToRemove.empty()) {
    RemovePred = [&Config, RemovePred](const std::unique_ptr<Section> &Sec) {
      return Config.ToRemove.matches(Sec->CanonicalName);
    };
  }

  if (Config.StripAll || Config.StripDebug) {
    // Remove all debug sections.
    RemovePred = [RemovePred](const std::unique_ptr<Section> &Sec) {
      if (Sec->Segname == "__DWARF")
        return true;

      return RemovePred(Sec);
    };
  }

  if (!Config.OnlySection.empty()) {
    // Overwrite RemovePred because --only-section takes priority.
    RemovePred = [&Config](const std::unique_ptr<Section> &Sec) {
      return !Config.OnlySection.matches(Sec->CanonicalName);
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
      Sym.Name = std::string(I->getValue());
  }

  auto RemovePred = [Config, &Obj](const std::unique_ptr<SymbolEntry> &N) {
    if (N->Referenced)
      return false;
    if (Config.StripAll)
      return true;
    if (Config.DiscardMode == DiscardType::All && !(N->n_type & MachO::N_EXT))
      return true;
    // This behavior is consistent with cctools' strip.
    if (Config.StripSwiftSymbols && (Obj.Header.Flags & MachO::MH_DYLDLINK) &&
        Obj.SwiftVersion && *Obj.SwiftVersion && N->isSwiftSymbol())
      return true;
    return false;
  };

  Obj.SymTable.removeSymbols(RemovePred);
}

template <typename LCType>
static void updateLoadCommandPayloadString(LoadCommand &LC, StringRef S) {
  assert(isLoadCommandWithPayloadString(LC) &&
         "unsupported load command encountered");

  uint32_t NewCmdsize = alignTo(sizeof(LCType) + S.size() + 1, 8);

  LC.MachOLoadCommand.load_command_data.cmdsize = NewCmdsize;
  LC.Payload.assign(NewCmdsize - sizeof(LCType), 0);
  std::copy(S.begin(), S.end(), LC.Payload.begin());
}

static LoadCommand buildRPathLoadCommand(StringRef Path) {
  LoadCommand LC;
  MachO::rpath_command RPathLC;
  RPathLC.cmd = MachO::LC_RPATH;
  RPathLC.path = sizeof(MachO::rpath_command);
  RPathLC.cmdsize = alignTo(sizeof(MachO::rpath_command) + Path.size() + 1, 8);
  LC.MachOLoadCommand.rpath_command_data = RPathLC;
  LC.Payload.assign(RPathLC.cmdsize - sizeof(MachO::rpath_command), 0);
  std::copy(Path.begin(), Path.end(), LC.Payload.begin());
  return LC;
}

static Error processLoadCommands(const CopyConfig &Config, Object &Obj) {
  // Remove RPaths.
  DenseSet<StringRef> RPathsToRemove(Config.RPathsToRemove.begin(),
                                     Config.RPathsToRemove.end());

  LoadCommandPred RemovePred = [&RPathsToRemove,
                                &Config](const LoadCommand &LC) {
    if (LC.MachOLoadCommand.load_command_data.cmd == MachO::LC_RPATH) {
      // When removing all RPaths we don't need to care
      // about what it contains
      if (Config.RemoveAllRpaths)
        return true;

      StringRef RPath = getPayloadString(LC);
      if (RPathsToRemove.count(RPath)) {
        RPathsToRemove.erase(RPath);
        return true;
      }
    }
    return false;
  };

  if (Error E = Obj.removeLoadCommands(RemovePred))
    return E;

  // Emit an error if the Mach-O binary does not contain an rpath path name
  // specified in -delete_rpath.
  for (StringRef RPath : Config.RPathsToRemove) {
    if (RPathsToRemove.count(RPath))
      return createStringError(errc::invalid_argument,
                               "no LC_RPATH load command with path: %s",
                               RPath.str().c_str());
  }

  DenseSet<StringRef> RPaths;

  // Get all existing RPaths.
  for (LoadCommand &LC : Obj.LoadCommands) {
    if (LC.MachOLoadCommand.load_command_data.cmd == MachO::LC_RPATH)
      RPaths.insert(getPayloadString(LC));
  }

  // Throw errors for invalid RPaths.
  for (const auto &OldNew : Config.RPathsToUpdate) {
    StringRef Old = OldNew.getFirst();
    StringRef New = OldNew.getSecond();
    if (RPaths.count(Old) == 0)
      return createStringError(errc::invalid_argument,
                               "no LC_RPATH load command with path: " + Old);
    if (RPaths.count(New) != 0)
      return createStringError(errc::invalid_argument,
                               "rpath " + New +
                                   " would create a duplicate load command");
  }

  // Update load commands.
  for (LoadCommand &LC : Obj.LoadCommands) {
    switch (LC.MachOLoadCommand.load_command_data.cmd) {
    case MachO::LC_ID_DYLIB:
      if (Config.SharedLibId)
        updateLoadCommandPayloadString<MachO::dylib_command>(
            LC, *Config.SharedLibId);
      break;

    case MachO::LC_RPATH: {
      StringRef RPath = getPayloadString(LC);
      StringRef NewRPath = Config.RPathsToUpdate.lookup(RPath);
      if (!NewRPath.empty())
        updateLoadCommandPayloadString<MachO::rpath_command>(LC, NewRPath);
      break;
    }

    // TODO: Add LC_REEXPORT_DYLIB, LC_LAZY_LOAD_DYLIB, and LC_LOAD_UPWARD_DYLIB
    // here once llvm-objcopy supports them.
    case MachO::LC_LOAD_DYLIB:
    case MachO::LC_LOAD_WEAK_DYLIB:
      StringRef InstallName = getPayloadString(LC);
      StringRef NewInstallName =
          Config.InstallNamesToUpdate.lookup(InstallName);
      if (!NewInstallName.empty())
        updateLoadCommandPayloadString<MachO::dylib_command>(LC,
                                                             NewInstallName);
      break;
    }
  }

  // Add new RPaths.
  for (StringRef RPath : Config.RPathToAdd) {
    if (RPaths.count(RPath) != 0)
      return createStringError(errc::invalid_argument,
                               "rpath " + RPath +
                                   " would create a duplicate load command");
    RPaths.insert(RPath);
    Obj.LoadCommands.push_back(buildRPathLoadCommand(RPath));
  }

  return Error::success();
}

static Error dumpSectionToFile(StringRef SecName, StringRef Filename,
                               Object &Obj) {
  for (LoadCommand &LC : Obj.LoadCommands)
    for (const std::unique_ptr<Section> &Sec : LC.Sections) {
      if (Sec->CanonicalName == SecName) {
        Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
            FileOutputBuffer::create(Filename, Sec->Content.size());
        if (!BufferOrErr)
          return BufferOrErr.takeError();
        std::unique_ptr<FileOutputBuffer> Buf = std::move(*BufferOrErr);
        llvm::copy(Sec->Content, Buf->getBufferStart());

        if (Error E = Buf->commit())
          return E;
        return Error::success();
      }
    }

  return createStringError(object_error::parse_failed, "section '%s' not found",
                           SecName.str().c_str());
}

static Error addSection(StringRef SecName, StringRef Filename, Object &Obj) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(Filename);
  if (!BufOrErr)
    return createFileError(Filename, errorCodeToError(BufOrErr.getError()));
  std::unique_ptr<MemoryBuffer> Buf = std::move(*BufOrErr);

  std::pair<StringRef, StringRef> Pair = SecName.split(',');
  StringRef TargetSegName = Pair.first;
  Section Sec(TargetSegName, Pair.second);
  Sec.Content = Obj.NewSectionsContents.save(Buf->getBuffer());

  // Add the a section into an existing segment.
  for (LoadCommand &LC : Obj.LoadCommands) {
    Optional<StringRef> SegName = LC.getSegmentName();
    if (SegName && SegName == TargetSegName) {
      uint64_t Addr = *LC.getSegmentVMAddr();
      for (const std::unique_ptr<Section> &S : LC.Sections)
        Addr = std::max(Addr, S->Addr + S->Size);
      LC.Sections.push_back(std::make_unique<Section>(Sec));
      LC.Sections.back()->Addr = Addr;
      return Error::success();
    }
  }

  // There's no segment named TargetSegName. Create a new load command and
  // Insert a new section into it.
  LoadCommand &NewSegment = Obj.addSegment(TargetSegName);
  NewSegment.Sections.push_back(std::make_unique<Section>(Sec));
  NewSegment.Sections.back()->Addr = *NewSegment.getSegmentVMAddr();
  return Error::success();
}

// isValidMachOCannonicalName returns success if Name is a MachO cannonical name
// ("<segment>,<section>") and lengths of both segment and section names are
// valid.
Error isValidMachOCannonicalName(StringRef Name) {
  if (Name.count(',') != 1)
    return createStringError(errc::invalid_argument,
                             "invalid section name '%s' (should be formatted "
                             "as '<segment name>,<section name>')",
                             Name.str().c_str());

  std::pair<StringRef, StringRef> Pair = Name.split(',');
  if (Pair.first.size() > 16)
    return createStringError(errc::invalid_argument,
                             "too long segment name: '%s'",
                             Pair.first.str().c_str());
  if (Pair.second.size() > 16)
    return createStringError(errc::invalid_argument,
                             "too long section name: '%s'",
                             Pair.second.str().c_str());
  return Error::success();
}

static Error handleArgs(const CopyConfig &Config, Object &Obj) {
  if (Config.AllowBrokenLinks || !Config.BuildIdLinkDir.empty() ||
      Config.BuildIdLinkInput || Config.BuildIdLinkOutput ||
      !Config.SplitDWO.empty() || !Config.SymbolsPrefix.empty() ||
      !Config.AllocSectionsPrefix.empty() || !Config.KeepSection.empty() ||
      Config.NewSymbolVisibility || !Config.SymbolsToGlobalize.empty() ||
      !Config.SymbolsToKeep.empty() || !Config.SymbolsToLocalize.empty() ||
      !Config.SymbolsToWeaken.empty() || !Config.SymbolsToKeepGlobal.empty() ||
      !Config.SectionsToRename.empty() ||
      !Config.UnneededSymbolsToRemove.empty() ||
      !Config.SetSectionAlignment.empty() || !Config.SetSectionFlags.empty() ||
      Config.ExtractDWO || Config.LocalizeHidden || Config.PreserveDates ||
      Config.StripAllGNU || Config.StripDWO || Config.StripNonAlloc ||
      Config.StripSections || Config.Weaken || Config.DecompressDebugSections ||
      Config.StripUnneeded || Config.DiscardMode == DiscardType::Locals ||
      !Config.SymbolsToAdd.empty() || Config.EntryExpr) {
    return createStringError(llvm::errc::invalid_argument,
                             "option not supported by llvm-objcopy for MachO");
  }

  // Dump sections before add/remove for compatibility with GNU objcopy.
  for (StringRef Flag : Config.DumpSection) {
    StringRef SectionName;
    StringRef FileName;
    std::tie(SectionName, FileName) = Flag.split('=');
    if (Error E = dumpSectionToFile(SectionName, FileName, Obj))
      return E;
  }

  if (Error E = removeSections(Config, Obj))
    return E;

  // Mark symbols to determine which symbols are still needed.
  if (Config.StripAll)
    markSymbols(Config, Obj);

  updateAndRemoveSymbols(Config, Obj);

  if (Config.StripAll)
    for (LoadCommand &LC : Obj.LoadCommands)
      for (std::unique_ptr<Section> &Sec : LC.Sections)
        Sec->Relocations.clear();

  for (const auto &Flag : Config.AddSection) {
    std::pair<StringRef, StringRef> SecPair = Flag.split("=");
    StringRef SecName = SecPair.first;
    StringRef File = SecPair.second;
    if (Error E = isValidMachOCannonicalName(SecName))
      return E;
    if (Error E = addSection(SecName, File, Obj))
      return E;
  }

  if (Error E = processLoadCommands(Config, Obj))
    return E;

  return Error::success();
}

Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::MachOObjectFile &In, Buffer &Out) {
  MachOReader Reader(In);
  Expected<std::unique_ptr<Object>> O = Reader.create();
  if (!O)
    return createFileError(Config.InputFilename, O.takeError());

  if (Error E = handleArgs(Config, **O))
    return createFileError(Config.InputFilename, std::move(E));

  // Page size used for alignment of segment sizes in Mach-O executables and
  // dynamic libraries.
  uint64_t PageSize;
  switch (In.getArch()) {
  case Triple::ArchType::arm:
  case Triple::ArchType::aarch64:
  case Triple::ArchType::aarch64_32:
    PageSize = 16384;
    break;
  default:
    PageSize = 4096;
  }

  MachOWriter Writer(**O, In.is64Bit(), In.isLittleEndian(), PageSize, Out);
  if (auto E = Writer.finalize())
    return E;
  return Writer.write();
}

Error executeObjcopyOnMachOUniversalBinary(CopyConfig &Config,
                                           const MachOUniversalBinary &In,
                                           Buffer &Out) {
  SmallVector<OwningBinary<Binary>, 2> Binaries;
  SmallVector<Slice, 2> Slices;
  for (const auto &O : In.objects()) {
    Expected<std::unique_ptr<Archive>> ArOrErr = O.getAsArchive();
    if (ArOrErr) {
      Expected<std::vector<NewArchiveMember>> NewArchiveMembersOrErr =
          createNewArchiveMembers(Config, **ArOrErr);
      if (!NewArchiveMembersOrErr)
        return NewArchiveMembersOrErr.takeError();
      Expected<std::unique_ptr<MemoryBuffer>> OutputBufferOrErr =
          writeArchiveToBuffer(*NewArchiveMembersOrErr,
                               (*ArOrErr)->hasSymbolTable(), (*ArOrErr)->kind(),
                               Config.DeterministicArchives,
                               (*ArOrErr)->isThin());
      if (!OutputBufferOrErr)
        return OutputBufferOrErr.takeError();
      Expected<std::unique_ptr<Binary>> BinaryOrErr =
          object::createBinary(**OutputBufferOrErr);
      if (!BinaryOrErr)
        return BinaryOrErr.takeError();
      Binaries.emplace_back(std::move(*BinaryOrErr),
                            std::move(*OutputBufferOrErr));
      Slices.emplace_back(*cast<Archive>(Binaries.back().getBinary()),
                          O.getCPUType(), O.getCPUSubType(),
                          O.getArchFlagName(), O.getAlign());
      continue;
    }
    // The methods getAsArchive, getAsObjectFile, getAsIRObject of the class
    // ObjectForArch return an Error in case of the type mismatch. We need to
    // check each in turn to see what kind of slice this is, so ignore errors
    // produced along the way.
    consumeError(ArOrErr.takeError());

    Expected<std::unique_ptr<MachOObjectFile>> ObjOrErr = O.getAsObjectFile();
    if (!ObjOrErr) {
      consumeError(ObjOrErr.takeError());
      return createStringError(std::errc::invalid_argument,
                               "slice for '%s' of the universal Mach-O binary "
                               "'%s' is not a Mach-O object or an archive",
                               O.getArchFlagName().c_str(),
                               Config.InputFilename.str().c_str());
    }
    std::string ArchFlagName = O.getArchFlagName();
    MemBuffer MB(ArchFlagName);
    if (Error E = executeObjcopyOnBinary(Config, **ObjOrErr, MB))
      return E;
    std::unique_ptr<WritableMemoryBuffer> OutputBuffer =
        MB.releaseMemoryBuffer();
    Expected<std::unique_ptr<Binary>> BinaryOrErr =
        object::createBinary(*OutputBuffer);
    if (!BinaryOrErr)
      return BinaryOrErr.takeError();
    Binaries.emplace_back(std::move(*BinaryOrErr), std::move(OutputBuffer));
    Slices.emplace_back(*cast<MachOObjectFile>(Binaries.back().getBinary()),
                        O.getAlign());
  }
  Expected<std::unique_ptr<MemoryBuffer>> B =
      writeUniversalBinaryToBuffer(Slices);
  if (!B)
    return B.takeError();
  if (Error E = Out.allocate((*B)->getBufferSize()))
    return E;
  memcpy(Out.getBufferStart(), (*B)->getBufferStart(), (*B)->getBufferSize());
  return Out.commit();
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
