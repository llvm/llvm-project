//===- llvm-ifs.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "ErrorCollector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/InterfaceStub/ELFObjHandler.h"
#include "llvm/InterfaceStub/IFSHandler.h"
#include "llvm/InterfaceStub/IFSStub.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TextAPI/InterfaceFile.h"
#include "llvm/TextAPI/TextAPIReader.h"
#include "llvm/TextAPI/TextAPIWriter.h"
#include <set>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::yaml;
using namespace llvm::MachO;
using namespace llvm::ifs;

#define DEBUG_TYPE "llvm-ifs"

namespace {
const VersionTuple IfsVersionCurrent(3, 0);

enum class FileFormat { IFS, ELF, TBD };
} // end anonymous namespace

cl::OptionCategory IfsCategory("Ifs Options");

// TODO: Use OptTable for option parsing in the future.
// Command line flags:
cl::list<std::string> InputFilePaths(cl::Positional, cl::desc("input"),
                                     cl::ZeroOrMore, cl::cat(IfsCategory));
cl::opt<FileFormat> InputFormat(
    "input-format", cl::desc("Specify the input file format"),
    cl::values(clEnumValN(FileFormat::IFS, "IFS", "Text based ELF stub file"),
               clEnumValN(FileFormat::ELF, "ELF", "ELF object file")),
    cl::cat(IfsCategory));
cl::opt<FileFormat> OutputFormat(
    "output-format", cl::desc("Specify the output file format **DEPRECATED**"),
    cl::values(clEnumValN(FileFormat::IFS, "IFS", "Text based ELF stub file"),
               clEnumValN(FileFormat::ELF, "ELF", "ELF stub file"),
               clEnumValN(FileFormat::TBD, "TBD", "Apple TBD text stub file")),
    cl::cat(IfsCategory));
cl::opt<std::string> OptArch("arch",
                             cl::desc("Specify the architecture, e.g. x86_64"),
                             cl::cat(IfsCategory));
cl::opt<IFSBitWidthType>
    OptBitWidth("bitwidth", cl::desc("Specify the bit width"),
                cl::values(clEnumValN(IFSBitWidthType::IFS32, "32", "32 bits"),
                           clEnumValN(IFSBitWidthType::IFS64, "64", "64 bits")),
                cl::cat(IfsCategory));
cl::opt<IFSEndiannessType> OptEndianness(
    "endianness", cl::desc("Specify the endianness"),
    cl::values(clEnumValN(IFSEndiannessType::Little, "little", "Little Endian"),
               clEnumValN(IFSEndiannessType::Big, "big", "Big Endian")),
    cl::cat(IfsCategory));
cl::opt<std::string> OptTargetTriple(
    "target", cl::desc("Specify the target triple, e.g. x86_64-linux-gnu"),
    cl::cat(IfsCategory));
cl::opt<std::string> OptTargetTripleHint(
    "hint-ifs-target",
    cl::desc("When --output-format is 'IFS', this flag will hint the expected "
             "target triple for IFS output"),
    cl::cat(IfsCategory));
cl::opt<bool> StripIFSArch(
    "strip-ifs-arch",
    cl::desc("Strip target architecture information away from IFS output"),
    cl::cat(IfsCategory));
cl::opt<bool> StripIFSBitWidth(
    "strip-ifs-bitwidth",
    cl::desc("Strip target bit width information away from IFS output"),
    cl::cat(IfsCategory));
cl::opt<bool> StripIFSEndiannessWidth(
    "strip-ifs-endianness",
    cl::desc("Strip target endianness information away from IFS output"),
    cl::cat(IfsCategory));
cl::opt<bool> StripIFSTarget(
    "strip-ifs-target",
    cl::desc("Strip all target information away from IFS output"),
    cl::cat(IfsCategory));
cl::opt<bool>
    StripUndefined("strip-undefined",
                   cl::desc("Strip undefined symbols from IFS output"),
                   cl::cat(IfsCategory));
cl::opt<bool> StripNeededLibs("strip-needed",
                              cl::desc("Strip needed libs from output"),
                              cl::cat(IfsCategory));

cl::opt<std::string>
    SoName("soname",
           cl::desc("Manually set the DT_SONAME entry of any emitted files"),
           cl::value_desc("name"), cl::cat(IfsCategory));
cl::opt<std::string> OutputFilePath("output",
                                    cl::desc("Output file **DEPRECATED**"),
                                    cl::cat(IfsCategory));
cl::alias OutputFilePathA("o", cl::desc("Alias for --output"),
                          cl::aliasopt(OutputFilePath), cl::cat(IfsCategory));
cl::opt<std::string> OutputELFFilePath("output-elf",
                                       cl::desc("Output path for ELF file"),
                                       cl::cat(IfsCategory));
cl::opt<std::string> OutputIFSFilePath("output-ifs",
                                       cl::desc("Output path for IFS file"),
                                       cl::cat(IfsCategory));
cl::opt<std::string> OutputTBDFilePath("output-tbd",
                                       cl::desc("Output path for TBD file"),
                                       cl::cat(IfsCategory));

cl::opt<bool> WriteIfChanged(
    "write-if-changed",
    cl::desc("Write the output file only if it is new or has changed."),
    cl::cat(IfsCategory));

static std::string getTypeName(IFSSymbolType Type) {
  switch (Type) {
  case IFSSymbolType::NoType:
    return "NoType";
  case IFSSymbolType::Func:
    return "Func";
  case IFSSymbolType::Object:
    return "Object";
  case IFSSymbolType::TLS:
    return "TLS";
  case IFSSymbolType::Unknown:
    return "Unknown";
  }
  llvm_unreachable("Unexpected ifs symbol type.");
}

static Expected<std::unique_ptr<IFSStub>> readInputFile(StringRef FilePath) {
  // Read in file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrError =
      MemoryBuffer::getFileOrSTDIN(FilePath, /*IsText=*/true);
  if (!BufOrError)
    return createStringError(BufOrError.getError(), "Could not open `%s`",
                             FilePath.data());

  std::unique_ptr<MemoryBuffer> FileReadBuffer = std::move(*BufOrError);
  ErrorCollector EC(/*UseFatalErrors=*/false);

  // First try to read as a binary (fails fast if not binary).
  if (InputFormat.getNumOccurrences() == 0 || InputFormat == FileFormat::ELF) {
    Expected<std::unique_ptr<IFSStub>> StubFromELF =
        readELFFile(FileReadBuffer->getMemBufferRef());
    if (StubFromELF) {
      (*StubFromELF)->IfsVersion = IfsVersionCurrent;
      return std::move(*StubFromELF);
    }
    EC.addError(StubFromELF.takeError(), "BinaryRead");
  }

  // Fall back to reading as a ifs.
  if (InputFormat.getNumOccurrences() == 0 || InputFormat == FileFormat::IFS) {
    Expected<std::unique_ptr<IFSStub>> StubFromIFS =
        readIFSFromBuffer(FileReadBuffer->getBuffer());
    if (StubFromIFS) {
      if ((*StubFromIFS)->IfsVersion > IfsVersionCurrent)
        EC.addError(
            createStringError(errc::not_supported,
                              "IFS version " +
                                  (*StubFromIFS)->IfsVersion.getAsString() +
                                  " is unsupported."),
            "ReadInputFile");
      else
        return std::move(*StubFromIFS);
    } else {
      EC.addError(StubFromIFS.takeError(), "YamlParse");
    }
  }

  // If both readers fail, build a new error that includes all information.
  EC.addError(createStringError(errc::not_supported,
                                "No file readers succeeded reading `%s` "
                                "(unsupported/malformed file?)",
                                FilePath.data()),
              "ReadInputFile");
  EC.escalateToFatal();
  return EC.makeError();
}

static int writeTbdStub(const Triple &T, const std::vector<IFSSymbol> &Symbols,
                        const StringRef Format, raw_ostream &Out) {

  auto PlatformTypeOrError =
      [](const llvm::Triple &T) -> llvm::Expected<llvm::MachO::PlatformType> {
    if (T.isMacOSX())
      return llvm::MachO::PLATFORM_MACOS;
    if (T.isTvOS())
      return llvm::MachO::PLATFORM_TVOS;
    if (T.isWatchOS())
      return llvm::MachO::PLATFORM_WATCHOS;
    // Note: put isiOS last because tvOS and watchOS are also iOS according
    // to the Triple.
    if (T.isiOS())
      return llvm::MachO::PLATFORM_IOS;

    return createStringError(errc::not_supported, "Invalid Platform.\n");
  }(T);

  if (!PlatformTypeOrError)
    return -1;

  PlatformType Plat = PlatformTypeOrError.get();
  TargetList Targets({Target(llvm::MachO::mapToArchitecture(T), Plat)});

  InterfaceFile File;
  File.setFileType(FileType::TBD_V3); // Only supporting v3 for now.
  File.addTargets(Targets);

  for (const auto &Symbol : Symbols) {
    auto Name = Symbol.Name;
    auto Kind = SymbolKind::GlobalSymbol;
    switch (Symbol.Type) {
    default:
    case IFSSymbolType::NoType:
      Kind = SymbolKind::GlobalSymbol;
      break;
    case IFSSymbolType::Object:
      Kind = SymbolKind::GlobalSymbol;
      break;
    case IFSSymbolType::Func:
      Kind = SymbolKind::GlobalSymbol;
      break;
    }
    if (Symbol.Weak)
      File.addSymbol(Kind, Name, Targets, SymbolFlags::WeakDefined);
    else
      File.addSymbol(Kind, Name, Targets);
  }

  SmallString<4096> Buffer;
  raw_svector_ostream OS(Buffer);
  if (Error Result = TextAPIWriter::writeToStream(OS, File))
    return -1;
  Out << OS.str();
  return 0;
}

static void fatalError(Error Err) {
  WithColor::defaultErrorHandler(std::move(Err));
  exit(1);
}

/// writeIFS() writes a Text-Based ELF stub to a file using the latest version
/// of the YAML parser.
static Error writeIFS(StringRef FilePath, IFSStub &Stub) {
  // Write IFS to memory first.
  std::string IFSStr;
  raw_string_ostream OutStr(IFSStr);
  Error YAMLErr = writeIFSToOutputStream(OutStr, Stub);
  if (YAMLErr)
    return YAMLErr;
  OutStr.flush();

  if (WriteIfChanged) {
    if (ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrError =
            MemoryBuffer::getFile(FilePath)) {
      // Compare IFS output with the existing IFS file. If unchanged, avoid changing the file.
      if ((*BufOrError)->getBuffer() == IFSStr)
        return Error::success();
    }
  }
  // Open IFS file for writing.
  std::error_code SysErr;
  raw_fd_ostream Out(FilePath, SysErr);
  if (SysErr)
    return createStringError(SysErr, "Couldn't open `%s` for writing",
                             FilePath.data());
  Out << IFSStr;
  return Error::success();
}

int main(int argc, char *argv[]) {
  // Parse arguments.
  cl::HideUnrelatedOptions({&IfsCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv);

  if (InputFilePaths.empty())
    InputFilePaths.push_back("-");

  // If input files are more than one, they can only be IFS files.
  if (InputFilePaths.size() > 1)
    InputFormat.setValue(FileFormat::IFS);

  // Attempt to merge input.
  IFSStub Stub;
  std::map<std::string, IFSSymbol> SymbolMap;
  std::string PreviousInputFilePath;
  for (const std::string &InputFilePath : InputFilePaths) {
    Expected<std::unique_ptr<IFSStub>> StubOrErr = readInputFile(InputFilePath);
    if (!StubOrErr)
      fatalError(StubOrErr.takeError());

    std::unique_ptr<IFSStub> TargetStub = std::move(StubOrErr.get());
    if (PreviousInputFilePath.empty()) {
      Stub.IfsVersion = TargetStub->IfsVersion;
      Stub.Target = TargetStub->Target;
      Stub.SoName = TargetStub->SoName;
      Stub.NeededLibs = TargetStub->NeededLibs;
    } else {
      if (Stub.IfsVersion != TargetStub->IfsVersion) {
        if (Stub.IfsVersion.getMajor() != IfsVersionCurrent.getMajor()) {
          WithColor::error()
              << "Interface Stub: IfsVersion Mismatch."
              << "\nFilenames: " << PreviousInputFilePath << " "
              << InputFilePath << "\nIfsVersion Values: " << Stub.IfsVersion
              << " " << TargetStub->IfsVersion << "\n";
          return -1;
        }
        if (TargetStub->IfsVersion > Stub.IfsVersion)
          Stub.IfsVersion = TargetStub->IfsVersion;
      }
      if (Stub.Target != TargetStub->Target && !TargetStub->Target.empty()) {
        WithColor::error() << "Interface Stub: Target Mismatch."
                           << "\nFilenames: " << PreviousInputFilePath << " "
                           << InputFilePath;
        return -1;
      }
      if (Stub.SoName != TargetStub->SoName) {
        WithColor::error() << "Interface Stub: SoName Mismatch."
                           << "\nFilenames: " << PreviousInputFilePath << " "
                           << InputFilePath
                           << "\nSoName Values: " << Stub.SoName << " "
                           << TargetStub->SoName << "\n";
        return -1;
      }
      if (Stub.NeededLibs != TargetStub->NeededLibs) {
        WithColor::error() << "Interface Stub: NeededLibs Mismatch."
                           << "\nFilenames: " << PreviousInputFilePath << " "
                           << InputFilePath << "\n";
        return -1;
      }
    }

    for (auto Symbol : TargetStub->Symbols) {
      auto SI = SymbolMap.find(Symbol.Name);
      if (SI == SymbolMap.end()) {
        SymbolMap.insert(
            std::pair<std::string, IFSSymbol>(Symbol.Name, Symbol));
        continue;
      }

      assert(Symbol.Name == SI->second.Name && "Symbol Names Must Match.");

      // Check conflicts:
      if (Symbol.Type != SI->second.Type) {
        WithColor::error() << "Interface Stub: Type Mismatch for "
                           << Symbol.Name << ".\nFilename: " << InputFilePath
                           << "\nType Values: " << getTypeName(SI->second.Type)
                           << " " << getTypeName(Symbol.Type) << "\n";

        return -1;
      }
      if (Symbol.Size != SI->second.Size) {
        WithColor::error() << "Interface Stub: Size Mismatch for "
                           << Symbol.Name << ".\nFilename: " << InputFilePath
                           << "\nSize Values: " << SI->second.Size << " "
                           << Symbol.Size << "\n";

        return -1;
      }
      if (Symbol.Weak != SI->second.Weak) {
        Symbol.Weak = false;
        continue;
      }
      // TODO: Not checking Warning. Will be dropped.
    }

    PreviousInputFilePath = InputFilePath;
  }

  if (Stub.IfsVersion != IfsVersionCurrent)
    if (Stub.IfsVersion.getMajor() != IfsVersionCurrent.getMajor()) {
      WithColor::error() << "Interface Stub: Bad IfsVersion: "
                         << Stub.IfsVersion << ", llvm-ifs supported version: "
                         << IfsVersionCurrent << ".\n";
      return -1;
    }

  for (auto &Entry : SymbolMap)
    Stub.Symbols.push_back(Entry.second);

  // Change SoName before emitting stubs.
  if (SoName.getNumOccurrences() == 1)
    Stub.SoName = SoName;
  Optional<IFSArch> OverrideArch;
  Optional<IFSEndiannessType> OverrideEndianness;
  Optional<IFSBitWidthType> OverrideBitWidth;
  Optional<std::string> OverrideTriple;
  if (OptArch.getNumOccurrences() == 1)
    OverrideArch = ELF::convertArchNameToEMachine(OptArch.getValue());
  if (OptEndianness.getNumOccurrences() == 1)
    OverrideEndianness = OptEndianness.getValue();
  if (OptBitWidth.getNumOccurrences() == 1)
    OverrideBitWidth = OptBitWidth.getValue();
  if (OptTargetTriple.getNumOccurrences() == 1)
    OverrideTriple = OptTargetTriple.getValue();
  Error OverrideError = overrideIFSTarget(
      Stub, OverrideArch, OverrideEndianness, OverrideBitWidth, OverrideTriple);
  if (OverrideError)
    fatalError(std::move(OverrideError));

  if (StripNeededLibs)
    Stub.NeededLibs.clear();

  if (OutputELFFilePath.getNumOccurrences() == 0 &&
      OutputIFSFilePath.getNumOccurrences() == 0 &&
      OutputTBDFilePath.getNumOccurrences() == 0) {
    if (OutputFormat.getNumOccurrences() == 0) {
      WithColor::error() << "at least one output should be specified.";
      return -1;
    }
  } else if (OutputFormat.getNumOccurrences() == 1) {
    WithColor::error() << "'--output-format' cannot be used with "
                          "'--output-{FILE_FORMAT}' options at the same time";
    return -1;
  }
  if (OutputFormat.getNumOccurrences() == 1) {
    // TODO: Remove OutputFormat flag in the next revision.
    WithColor::warning() << "--output-format option is deprecated, please use "
                            "--output-{FILE_FORMAT} options instead\n";
    switch (OutputFormat.getValue()) {
    case FileFormat::TBD: {
      std::error_code SysErr;
      raw_fd_ostream Out(OutputFilePath, SysErr);
      if (SysErr) {
        WithColor::error() << "Couldn't open " << OutputFilePath
                           << " for writing.\n";
        return -1;
      }
      if (!Stub.Target.Triple) {
        WithColor::error()
            << "Triple should be defined when output format is TBD";
        return -1;
      }
      return writeTbdStub(llvm::Triple(Stub.Target.Triple.getValue()),
                          Stub.Symbols, "TBD", Out);
    }
    case FileFormat::IFS: {
      Stub.IfsVersion = IfsVersionCurrent;
      if (InputFormat.getValue() == FileFormat::ELF &&
          OptTargetTripleHint.getNumOccurrences() == 1) {
        std::error_code HintEC(1, std::generic_category());
        IFSTarget HintTarget = parseTriple(OptTargetTripleHint);
        if (Stub.Target.Arch.getValue() != HintTarget.Arch.getValue())
          fatalError(make_error<StringError>(
              "Triple hint does not match the actual architecture", HintEC));
        if (Stub.Target.Endianness.getValue() !=
            HintTarget.Endianness.getValue())
          fatalError(make_error<StringError>(
              "Triple hint does not match the actual endianness", HintEC));
        if (Stub.Target.BitWidth.getValue() != HintTarget.BitWidth.getValue())
          fatalError(make_error<StringError>(
              "Triple hint does not match the actual bit width", HintEC));

        stripIFSTarget(Stub, true, false, false, false);
        Stub.Target.Triple = OptTargetTripleHint.getValue();
      } else {
        stripIFSTarget(Stub, StripIFSTarget, StripIFSArch,
                       StripIFSEndiannessWidth, StripIFSBitWidth);
      }
      if (StripUndefined)
        stripIFSUndefinedSymbols(Stub);
      Error IFSWriteError = writeIFS(OutputFilePath.getValue(), Stub);
      if (IFSWriteError)
        fatalError(std::move(IFSWriteError));
      break;
    }
    case FileFormat::ELF: {
      Error TargetError = validateIFSTarget(Stub, true);
      if (TargetError)
        fatalError(std::move(TargetError));
      Error BinaryWriteError =
          writeBinaryStub(OutputFilePath, Stub, WriteIfChanged);
      if (BinaryWriteError)
        fatalError(std::move(BinaryWriteError));
      break;
    }
    }
  } else {
    // Check if output path for individual format.
    if (OutputELFFilePath.getNumOccurrences() == 1) {
      Error TargetError = validateIFSTarget(Stub, true);
      if (TargetError)
        fatalError(std::move(TargetError));
      Error BinaryWriteError =
          writeBinaryStub(OutputELFFilePath, Stub, WriteIfChanged);
      if (BinaryWriteError)
        fatalError(std::move(BinaryWriteError));
    }
    if (OutputIFSFilePath.getNumOccurrences() == 1) {
      Stub.IfsVersion = IfsVersionCurrent;
      if (InputFormat.getValue() == FileFormat::ELF &&
          OptTargetTripleHint.getNumOccurrences() == 1) {
        std::error_code HintEC(1, std::generic_category());
        IFSTarget HintTarget = parseTriple(OptTargetTripleHint);
        if (Stub.Target.Arch.getValue() != HintTarget.Arch.getValue())
          fatalError(make_error<StringError>(
              "Triple hint does not match the actual architecture", HintEC));
        if (Stub.Target.Endianness.getValue() !=
            HintTarget.Endianness.getValue())
          fatalError(make_error<StringError>(
              "Triple hint does not match the actual endianness", HintEC));
        if (Stub.Target.BitWidth.getValue() != HintTarget.BitWidth.getValue())
          fatalError(make_error<StringError>(
              "Triple hint does not match the actual bit width", HintEC));

        stripIFSTarget(Stub, true, false, false, false);
        Stub.Target.Triple = OptTargetTripleHint.getValue();
      } else {
        stripIFSTarget(Stub, StripIFSTarget, StripIFSArch,
                       StripIFSEndiannessWidth, StripIFSBitWidth);
      }
      if (StripUndefined)
        stripIFSUndefinedSymbols(Stub);
      Error IFSWriteError = writeIFS(OutputIFSFilePath.getValue(), Stub);
      if (IFSWriteError)
        fatalError(std::move(IFSWriteError));
    }
    if (OutputTBDFilePath.getNumOccurrences() == 1) {
      std::error_code SysErr;
      raw_fd_ostream Out(OutputTBDFilePath, SysErr);
      if (SysErr) {
        WithColor::error() << "Couldn't open " << OutputTBDFilePath
                           << " for writing.\n";
        return -1;
      }
      if (!Stub.Target.Triple) {
        WithColor::error()
            << "Triple should be defined when output format is TBD";
        return -1;
      }
      return writeTbdStub(llvm::Triple(Stub.Target.Triple.getValue()),
                          Stub.Symbols, "TBD", Out);
    }
  }
  return 0;
}
