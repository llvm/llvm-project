//===- DlltoolDriver.cpp - dlltool.exe-compatible driver ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines an interface to a dlltool.exe-compatible driver.
//
//===----------------------------------------------------------------------===//

#include "llvm/ToolDrivers/llvm-dlltool/DlltoolDriver.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/COFFImportFile.h"
#include "llvm/Object/COFFModuleDefinition.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"

#include <optional>
#include <vector>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::COFF;

namespace {

#define OPTTABLE_STR_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_STR_TABLE_CODE

enum {
  OPT_INVALID = 0,
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

using namespace llvm::opt;
static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

class DllOptTable : public opt::GenericOptTable {
public:
  DllOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable,
                             false) {}
};

// Opens a file. Path has to be resolved already.
std::unique_ptr<MemoryBuffer> openFile(const Twine &Path) {
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> MB = MemoryBuffer::getFile(Path);

  if (std::error_code EC = MB.getError()) {
    llvm::errs() << "cannot open file " << Path << ": " << EC.message() << "\n";
    return nullptr;
  }

  return std::move(*MB);
}

MachineTypes getEmulation(StringRef S) {
  return StringSwitch<MachineTypes>(S)
      .Case("i386", IMAGE_FILE_MACHINE_I386)
      .Case("i386:x86-64", IMAGE_FILE_MACHINE_AMD64)
      .Case("arm", IMAGE_FILE_MACHINE_ARMNT)
      .Case("arm64", IMAGE_FILE_MACHINE_ARM64)
      .Case("arm64ec", IMAGE_FILE_MACHINE_ARM64EC)
      .Case("r4000", IMAGE_FILE_MACHINE_R4000)
      .Default(IMAGE_FILE_MACHINE_UNKNOWN);
}

MachineTypes getMachine(Triple T) {
  switch (T.getArch()) {
  case Triple::x86:
    return COFF::IMAGE_FILE_MACHINE_I386;
  case Triple::x86_64:
    return COFF::IMAGE_FILE_MACHINE_AMD64;
  case Triple::arm:
    return COFF::IMAGE_FILE_MACHINE_ARMNT;
  case Triple::aarch64:
    return T.isWindowsArm64EC() ? COFF::IMAGE_FILE_MACHINE_ARM64EC
                                : COFF::IMAGE_FILE_MACHINE_ARM64;
  case Triple::mipsel:
    return COFF::IMAGE_FILE_MACHINE_R4000;
  default:
    return COFF::IMAGE_FILE_MACHINE_UNKNOWN;
  }
}

MachineTypes getDefaultMachine() {
  return getMachine(Triple(sys::getDefaultTargetTriple()));
}

std::optional<std::string> getPrefix(StringRef Argv0) {
  StringRef ProgName = llvm::sys::path::stem(Argv0);
  // x86_64-w64-mingw32-dlltool -> x86_64-w64-mingw32
  // llvm-dlltool -> None
  // aarch64-w64-mingw32-llvm-dlltool-10.exe -> aarch64-w64-mingw32
  ProgName = ProgName.rtrim("0123456789.-");
  if (!ProgName.consume_back_insensitive("dlltool"))
    return std::nullopt;
  ProgName.consume_back_insensitive("llvm-");
  ProgName.consume_back_insensitive("-");
  return ProgName.str();
}

bool parseModuleDefinition(StringRef DefFileName, MachineTypes Machine,
                           bool AddUnderscores,
                           std::vector<COFFShortExport> &Exports,
                           std::string &OutputFile) {
  std::unique_ptr<MemoryBuffer> MB = openFile(DefFileName);
  if (!MB)
    return false;

  if (!MB->getBufferSize()) {
    llvm::errs() << "definition file empty\n";
    return false;
  }

  Expected<COFFModuleDefinition> Def = parseCOFFModuleDefinition(
      *MB, Machine, /*MingwDef=*/true, AddUnderscores);
  if (!Def) {
    llvm::errs() << "error parsing definition\n"
                 << errorToErrorCode(Def.takeError()).message() << "\n";
    return false;
  }

  if (OutputFile.empty())
    OutputFile = std::move(Def->OutputFile);

  // If ExtName is set (if the "ExtName = Name" syntax was used), overwrite
  // Name with ExtName and clear ExtName. When only creating an import
  // library and not linking, the internal name is irrelevant. This avoids
  // cases where writeImportLibrary tries to transplant decoration from
  // symbol decoration onto ExtName.
  for (COFFShortExport &E : Def->Exports) {
    if (!E.ExtName.empty()) {
      E.Name = E.ExtName;
      E.ExtName.clear();
    }
  }

  Exports = std::move(Def->Exports);
  return true;
}

int printError(llvm::Error E, Twine File) {
  if (!E)
    return 0;
  handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
    llvm::errs() << "error opening " << File << ": " << EIB.message() << "\n";
  });
  return 1;
}

template <typename Callable>
int forEachCoff(object::Archive &Archive, StringRef Name, Callable Callback) {
  Error Err = Error::success();
  for (auto &C : Archive.children(Err)) {
    Expected<StringRef> NameOrErr = C.getName();
    if (!NameOrErr)
      return printError(NameOrErr.takeError(), Name);
    StringRef Name = *NameOrErr;

    Expected<MemoryBufferRef> ChildMB = C.getMemoryBufferRef();
    if (!ChildMB)
      return printError(ChildMB.takeError(), Name);

    if (identify_magic(ChildMB->getBuffer()) == file_magic::coff_object) {
      auto Obj = object::COFFObjectFile::create(*ChildMB);
      if (!Obj)
        return printError(Obj.takeError(), Name);
      if (!Callback(*Obj->get(), Name))
        return 1;
    }
  }
  if (Err)
    return printError(std::move(Err), Name);
  return 0;
}

// To find the named of the imported DLL from an import library, we can either
// inspect the object files that form the import table entries, or we could
// just look at the archive member names, for MSVC style import libraries.
// Looking at the archive member names doesn't work for GNU style import
// libraries though, while inspecting the import table entries works for
// both. (MSVC style import libraries contain a couple regular object files
// for the header/trailers.)
//
// This implementation does the same as GNU dlltool does; look at the
// content of ".idata$7" sections, or for MSVC style libraries, look
// at ".idata$6" sections.
//
// For GNU style import libraries, there are also other data chunks in sections
// named ".idata$7" (entries to the IAT or ILT); these are distinguished
// by seeing that they contain relocations. (They also look like an empty
// string when looking for null termination.)
//
// Alternatively, we could do things differently - look for any .idata$2
// section; this would be import directory entries. At offset 0xc in them
// there is the RVA of the import DLL name; look for a relocation at this
// spot and locate the symbol that it points at. That symbol may either
// be within the same object file (in the case of MSVC style import libraries)
// or another object file (in the case of GNU import libraries).
bool identifyImportName(const COFFObjectFile &Obj, StringRef ObjName,
                        std::vector<StringRef> &Names, bool IsMsStyleImplib) {
  StringRef TargetName = IsMsStyleImplib ? ".idata$6" : ".idata$7";
  for (const auto &S : Obj.sections()) {
    Expected<StringRef> NameOrErr = S.getName();
    if (!NameOrErr) {
      printError(NameOrErr.takeError(), ObjName);
      return false;
    }
    StringRef Name = *NameOrErr;
    if (Name != TargetName)
      continue;

    // GNU import libraries contain .idata$7 section in the per function
    // objects too, but they contain relocations.
    if (!IsMsStyleImplib && !S.relocations().empty())
      continue;

    Expected<StringRef> ContentsOrErr = S.getContents();
    if (!ContentsOrErr) {
      printError(ContentsOrErr.takeError(), ObjName);
      return false;
    }
    StringRef Contents = *ContentsOrErr;
    Contents = Contents.substr(0, Contents.find('\0'));
    if (Contents.empty())
      continue;
    Names.push_back(Contents);
    return true;
  }
  return true;
}

int doIdentify(StringRef File, bool IdentifyStrict) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeBuf = MemoryBuffer::getFile(
      File, /*IsText=*/false, /*RequiredNullTerminator=*/false);
  if (!MaybeBuf)
    return printError(errorCodeToError(MaybeBuf.getError()), File);
  if (identify_magic(MaybeBuf.get()->getBuffer()) != file_magic::archive) {
    llvm::errs() << File << " is not a library\n";
    return 1;
  }

  std::unique_ptr<MemoryBuffer> B = std::move(MaybeBuf.get());
  Error Err = Error::success();
  object::Archive Archive(B->getMemBufferRef(), Err);
  if (Err)
    return printError(std::move(Err), B->getBufferIdentifier());

  bool IsMsStyleImplib = false;
  for (const auto &S : Archive.symbols()) {
    if (S.getName() == "__NULL_IMPORT_DESCRIPTOR") {
      IsMsStyleImplib = true;
      break;
    }
  }
  std::vector<StringRef> Names;
  if (forEachCoff(Archive, B->getBufferIdentifier(),
                  [&](const COFFObjectFile &Obj, StringRef ObjName) -> bool {
                    return identifyImportName(Obj, ObjName, Names,
                                              IsMsStyleImplib);
                  }))
    return 1;

  if (Names.empty()) {
    llvm::errs() << "No DLL import name found in " << File << "\n";
    return 1;
  }
  if (Names.size() > 1 && IdentifyStrict) {
    llvm::errs() << File << "contains imports for two or more DLLs\n";
    return 1;
  }

  for (StringRef S : Names)
    llvm::outs() << S << "\n";

  return 0;
}

} // namespace

int llvm::dlltoolDriverMain(llvm::ArrayRef<const char *> ArgsArr) {
  DllOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;
  llvm::opt::InputArgList Args =
      Table.ParseArgs(ArgsArr.slice(1), MissingIndex, MissingCount);
  if (MissingCount) {
    llvm::errs() << Args.getArgString(MissingIndex) << ": missing argument\n";
    return 1;
  }

  // Handle when no input or output is specified
  if (Args.hasArgNoClaim(OPT_INPUT) ||
      (!Args.hasArgNoClaim(OPT_d) && !Args.hasArgNoClaim(OPT_l) &&
       !Args.hasArgNoClaim(OPT_I))) {
    Table.printHelp(outs(), "llvm-dlltool [options] file...", "llvm-dlltool",
                    false);
    llvm::outs()
        << "\nTARGETS: i386, i386:x86-64, arm, arm64, arm64ec, r4000\n";
    return 1;
  }

  for (auto *Arg : Args.filtered(OPT_UNKNOWN))
    llvm::errs() << "ignoring unknown argument: " << Arg->getAsString(Args)
                 << "\n";

  if (Args.hasArg(OPT_I)) {
    return doIdentify(Args.getLastArg(OPT_I)->getValue(),
                      Args.hasArg(OPT_identify_strict));
  }

  if (!Args.hasArg(OPT_d)) {
    llvm::errs() << "no definition file specified\n";
    return 1;
  }

  COFF::MachineTypes Machine = getDefaultMachine();
  if (std::optional<std::string> Prefix = getPrefix(ArgsArr[0])) {
    Triple T(*Prefix);
    if (T.getArch() != Triple::UnknownArch)
      Machine = getMachine(T);
  }
  if (auto *Arg = Args.getLastArg(OPT_m))
    Machine = getEmulation(Arg->getValue());

  if (Machine == IMAGE_FILE_MACHINE_UNKNOWN) {
    llvm::errs() << "unknown target\n";
    return 1;
  }

  bool AddUnderscores = !Args.hasArg(OPT_no_leading_underscore);

  std::string OutputFile;
  if (auto *Arg = Args.getLastArg(OPT_D))
    OutputFile = Arg->getValue();

  std::vector<COFFShortExport> Exports, NativeExports;

  if (Args.hasArg(OPT_N)) {
    if (!isArm64EC(Machine)) {
      llvm::errs() << "native .def file is supported only on arm64ec target\n";
      return 1;
    }
    if (!parseModuleDefinition(Args.getLastArg(OPT_N)->getValue(),
                               IMAGE_FILE_MACHINE_ARM64, AddUnderscores,
                               NativeExports, OutputFile))
      return 1;
  }

  if (!parseModuleDefinition(Args.getLastArg(OPT_d)->getValue(), Machine,
                             AddUnderscores, Exports, OutputFile))
    return 1;

  if (OutputFile.empty()) {
    llvm::errs() << "no DLL name specified\n";
    return 1;
  }

  if (Machine == IMAGE_FILE_MACHINE_I386 && Args.hasArg(OPT_k)) {
    for (COFFShortExport &E : Exports) {
      if (!E.ImportName.empty() || (!E.Name.empty() && E.Name[0] == '?'))
        continue;
      E.SymbolName = E.Name;
      // Trim off the trailing decoration. Symbols will always have a
      // starting prefix here (either _ for cdecl/stdcall, @ for fastcall
      // or ? for C++ functions). Vectorcall functions won't have any
      // fixed prefix, but the function base name will still be at least
      // one char.
      E.Name = E.Name.substr(0, E.Name.find('@', 1));
      // By making sure E.SymbolName != E.Name for decorated symbols,
      // writeImportLibrary writes these symbols with the type
      // IMPORT_NAME_UNDECORATE.
    }
  }

  std::string Path = std::string(Args.getLastArgValue(OPT_l));
  if (!Path.empty()) {
    if (Error E = writeImportLibrary(OutputFile, Path, Exports, Machine,
                                     /*MinGW=*/true, NativeExports)) {
      handleAllErrors(std::move(E), [&](const ErrorInfoBase &EI) {
        llvm::errs() << EI.message() << "\n";
      });
      return 1;
    }
  }
  return 0;
}
