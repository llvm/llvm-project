//===- llvm-ifs.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
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
#include "llvm/TextAPI/MachO/InterfaceFile.h"
#include "llvm/TextAPI/MachO/TextAPIReader.h"
#include "llvm/TextAPI/MachO/TextAPIWriter.h"
#include <set>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::yaml;
using namespace llvm::MachO;

#define DEBUG_TYPE "llvm-ifs"

namespace {
const VersionTuple IFSVersionCurrent(2, 0);
} // end anonymous namespace

static cl::opt<std::string> Action("action", cl::desc("<llvm-ifs action>"),
                                   cl::value_desc("write-ifs | write-bin"),
                                   cl::init("write-ifs"));

static cl::opt<std::string> ForceFormat("force-format",
                                        cl::desc("<force object format>"),
                                        cl::value_desc("ELF | TBD"),
                                        cl::init(""));

static cl::list<std::string> InputFilenames(cl::Positional,
                                            cl::desc("<input ifs files>"),
                                            cl::ZeroOrMore);

static cl::opt<std::string> OutputFilename("o", cl::desc("<output file>"),
                                           cl::value_desc("path"));

enum class IFSSymbolType {
  NoType = 0,
  Object,
  Func,
  // Type information is 4 bits, so 16 is safely out of range.
  Unknown = 16,
};

static std::string getTypeName(IFSSymbolType Type) {
  switch (Type) {
  case IFSSymbolType::NoType:
    return "NoType";
  case IFSSymbolType::Func:
    return "Func";
  case IFSSymbolType::Object:
    return "Object";
  case IFSSymbolType::Unknown:
    return "Unknown";
  }
  llvm_unreachable("Unexpected ifs symbol type.");
}

struct IFSSymbol {
  IFSSymbol() = default;
  IFSSymbol(std::string SymbolName) : Name(SymbolName) {}
  std::string Name;
  uint64_t Size;
  IFSSymbolType Type;
  bool Weak;
  Optional<std::string> Warning;
  bool operator<(const IFSSymbol &RHS) const { return Name < RHS.Name; }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(IFSSymbol)

namespace llvm {
namespace yaml {
/// YAML traits for IFSSymbolType.
template <> struct ScalarEnumerationTraits<IFSSymbolType> {
  static void enumeration(IO &IO, IFSSymbolType &SymbolType) {
    IO.enumCase(SymbolType, "NoType", IFSSymbolType::NoType);
    IO.enumCase(SymbolType, "Func", IFSSymbolType::Func);
    IO.enumCase(SymbolType, "Object", IFSSymbolType::Object);
    IO.enumCase(SymbolType, "Unknown", IFSSymbolType::Unknown);
    // Treat other symbol types as noise, and map to Unknown.
    if (!IO.outputting() && IO.matchEnumFallback())
      SymbolType = IFSSymbolType::Unknown;
  }
};

/// YAML traits for IFSSymbol.
template <> struct MappingTraits<IFSSymbol> {
  static void mapping(IO &IO, IFSSymbol &Symbol) {
    IO.mapRequired("Name", Symbol.Name);
    IO.mapRequired("Type", Symbol.Type);
    // The need for symbol size depends on the symbol type.
    if (Symbol.Type == IFSSymbolType::NoType)
      IO.mapOptional("Size", Symbol.Size, (uint64_t)0);
    else if (Symbol.Type == IFSSymbolType::Func)
      Symbol.Size = 0;
    else
      IO.mapRequired("Size", Symbol.Size);
    IO.mapOptional("Weak", Symbol.Weak, false);
    IO.mapOptional("Warning", Symbol.Warning);
  }

  // Compacts symbol information into a single line.
  static const bool flow = true;
};

} // namespace yaml
} // namespace llvm

// A cumulative representation of ELF stubs.
// Both textual and binary stubs will read into and write from this object.
class IFSStub {
  // TODO: Add support for symbol versioning.
public:
  VersionTuple IfsVersion;
  std::string Triple;
  std::string ObjectFileFormat;
  Optional<std::string> SOName;
  std::vector<std::string> NeededLibs;
  std::vector<IFSSymbol> Symbols;

  IFSStub() = default;
  IFSStub(const IFSStub &Stub)
      : IfsVersion(Stub.IfsVersion), Triple(Stub.Triple),
        ObjectFileFormat(Stub.ObjectFileFormat), SOName(Stub.SOName),
        NeededLibs(Stub.NeededLibs), Symbols(Stub.Symbols) {}
  IFSStub(IFSStub &&Stub)
      : IfsVersion(std::move(Stub.IfsVersion)), Triple(std::move(Stub.Triple)),
        ObjectFileFormat(std::move(Stub.ObjectFileFormat)),
        SOName(std::move(Stub.SOName)), NeededLibs(std::move(Stub.NeededLibs)),
        Symbols(std::move(Stub.Symbols)) {}
};

namespace llvm {
namespace yaml {
/// YAML traits for IFSStub objects.
template <> struct MappingTraits<IFSStub> {
  static void mapping(IO &IO, IFSStub &Stub) {
    if (!IO.mapTag("!experimental-ifs-v2", true))
      IO.setError("Not a .ifs YAML file.");

    auto OldContext = IO.getContext();
    IO.setContext(&Stub);
    IO.mapRequired("IfsVersion", Stub.IfsVersion);
    IO.mapOptional("Triple", Stub.Triple);
    IO.mapOptional("ObjectFileFormat", Stub.ObjectFileFormat);
    IO.mapOptional("SOName", Stub.SOName);
    IO.mapOptional("NeededLibs", Stub.NeededLibs);
    IO.mapRequired("Symbols", Stub.Symbols);
    IO.setContext(&OldContext);
  }
};
} // namespace yaml
} // namespace llvm

static Expected<std::unique_ptr<IFSStub>> readInputFile(StringRef FilePath) {
  // Read in file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrError =
      MemoryBuffer::getFileOrSTDIN(FilePath);
  if (!BufOrError)
    return createStringError(BufOrError.getError(), "Could not open `%s`",
                             FilePath.data());

  std::unique_ptr<MemoryBuffer> FileReadBuffer = std::move(*BufOrError);
  yaml::Input YamlIn(FileReadBuffer->getBuffer());
  std::unique_ptr<IFSStub> Stub(new IFSStub());
  YamlIn >> *Stub;

  if (std::error_code Err = YamlIn.error())
    return createStringError(Err, "Failed reading Interface Stub File.");

  if (Stub->IfsVersion > IFSVersionCurrent)
    return make_error<StringError>(
        "IFS version " + Stub->IfsVersion.getAsString() + " is unsupported.",
        std::make_error_code(std::errc::invalid_argument));

  return std::move(Stub);
}

static int writeTbdStub(const Triple &T, const std::vector<IFSSymbol> &Symbols,
                        const StringRef Format, raw_ostream &Out) {

  auto PlatformKindOrError =
      [](const llvm::Triple &T) -> llvm::Expected<llvm::MachO::PlatformKind> {
    if (T.isMacOSX())
      return llvm::MachO::PlatformKind::macOS;
    if (T.isTvOS())
      return llvm::MachO::PlatformKind::tvOS;
    if (T.isWatchOS())
      return llvm::MachO::PlatformKind::watchOS;
    // Note: put isiOS last because tvOS and watchOS are also iOS according
    // to the Triple.
    if (T.isiOS())
      return llvm::MachO::PlatformKind::iOS;

    // TODO: Add an option for ForceTriple, but keep ForceFormat for now.
    if (ForceFormat == "TBD")
      return llvm::MachO::PlatformKind::macOS;

    return createStringError(errc::not_supported, "Invalid Platform.\n");
  }(T);

  if (!PlatformKindOrError)
    return -1;

  PlatformKind Plat = PlatformKindOrError.get();
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

static int writeElfStub(const Triple &T, const std::vector<IFSSymbol> &Symbols,
                        const StringRef Format, raw_ostream &Out) {
  SmallString<0> Storage;
  Storage.clear();
  raw_svector_ostream OS(Storage);

  OS << "--- !ELF\n";
  OS << "FileHeader:\n";
  OS << "  Class:           ELFCLASS";
  OS << (T.isArch64Bit() ? "64" : "32");
  OS << "\n";
  OS << "  Data:            ELFDATA2";
  OS << (T.isLittleEndian() ? "LSB" : "MSB");
  OS << "\n";
  OS << "  Type:            ET_DYN\n";
  OS << "  Machine:         "
     << llvm::StringSwitch<llvm::StringRef>(T.getArchName())
            .Case("x86_64", "EM_X86_64")
            .Case("i386", "EM_386")
            .Case("i686", "EM_386")
            .Case("aarch64", "EM_AARCH64")
            .Case("amdgcn", "EM_AMDGPU")
            .Case("r600", "EM_AMDGPU")
            .Case("arm", "EM_ARM")
            .Case("thumb", "EM_ARM")
            .Case("avr", "EM_AVR")
            .Case("mips", "EM_MIPS")
            .Case("mipsel", "EM_MIPS")
            .Case("mips64", "EM_MIPS")
            .Case("mips64el", "EM_MIPS")
            .Case("msp430", "EM_MSP430")
            .Case("ppc", "EM_PPC")
            .Case("ppc64", "EM_PPC64")
            .Case("ppc64le", "EM_PPC64")
            .Case("x86", T.isOSIAMCU() ? "EM_IAMCU" : "EM_386")
            .Case("x86_64", "EM_X86_64")
            .Default("EM_NONE")
     << "\nSections:"
     << "\n  - Name:            .text"
     << "\n    Type:            SHT_PROGBITS"
     << "\n  - Name:            .data"
     << "\n    Type:            SHT_PROGBITS"
     << "\n  - Name:            .rodata"
     << "\n    Type:            SHT_PROGBITS"
     << "\nSymbols:\n";
  for (const auto &Symbol : Symbols) {
    OS << "  - Name:            " << Symbol.Name << "\n"
       << "    Type:            STT_";
    switch (Symbol.Type) {
    default:
    case IFSSymbolType::NoType:
      OS << "NOTYPE";
      break;
    case IFSSymbolType::Object:
      OS << "OBJECT";
      break;
    case IFSSymbolType::Func:
      OS << "FUNC";
      break;
    }
    OS << "\n    Section:         .text"
       << "\n    Binding:         STB_" << (Symbol.Weak ? "WEAK" : "GLOBAL")
       << "\n";
  }
  OS << "...\n";

  std::string YamlStr = std::string(OS.str());

  // Only or debugging. Not an offical format.
  LLVM_DEBUG({
    if (ForceFormat == "ELFOBJYAML") {
      Out << YamlStr;
      return 0;
    }
  });

  yaml::Input YIn(YamlStr);
  auto ErrHandler = [](const Twine &Msg) {
    WithColor::error(errs(), "llvm-ifs") << Msg << "\n";
  };
  return convertYAML(YIn, Out, ErrHandler) ? 0 : 1;
}

static int writeIfso(const IFSStub &Stub, bool IsWriteIfs, raw_ostream &Out) {
  if (IsWriteIfs) {
    yaml::Output YamlOut(Out, NULL, /*WrapColumn =*/0);
    YamlOut << const_cast<IFSStub &>(Stub);
    return 0;
  }

  std::string ObjectFileFormat =
      ForceFormat.empty() ? Stub.ObjectFileFormat : ForceFormat;

  if (ObjectFileFormat == "ELF" || ForceFormat == "ELFOBJYAML")
    return writeElfStub(llvm::Triple(Stub.Triple), Stub.Symbols,
                        Stub.ObjectFileFormat, Out);
  if (ObjectFileFormat == "TBD")
    return writeTbdStub(llvm::Triple(Stub.Triple), Stub.Symbols,
                        Stub.ObjectFileFormat, Out);

  WithColor::error()
      << "Invalid ObjectFileFormat: Only ELF and TBD are supported.\n";
  return -1;
}

// TODO: Drop ObjectFileFormat, it can be subsumed from the triple.
// New Interface Stubs Yaml Format:
// --- !experimental-ifs-v2
// IfsVersion: 2.0
// Triple:          <llvm triple>
// ObjectFileFormat: <ELF | others not yet supported>
// Symbols:
//   _ZSymbolName: { Type: <type> }
// ...

int main(int argc, char *argv[]) {
  // Parse arguments.
  cl::ParseCommandLineOptions(argc, argv);

  if (InputFilenames.empty())
    InputFilenames.push_back("-");

  IFSStub Stub;
  std::map<std::string, IFSSymbol> SymbolMap;

  std::string PreviousInputFilePath = "";
  for (const std::string &InputFilePath : InputFilenames) {
    Expected<std::unique_ptr<IFSStub>> StubOrErr = readInputFile(InputFilePath);
    if (!StubOrErr) {
      WithColor::error() << StubOrErr.takeError() << "\n";
      return -1;
    }
    std::unique_ptr<IFSStub> TargetStub = std::move(StubOrErr.get());

    if (Stub.Triple.empty()) {
      PreviousInputFilePath = InputFilePath;
      Stub.IfsVersion = TargetStub->IfsVersion;
      Stub.Triple = TargetStub->Triple;
      Stub.ObjectFileFormat = TargetStub->ObjectFileFormat;
      Stub.SOName = TargetStub->SOName;
      Stub.NeededLibs = TargetStub->NeededLibs;
    } else {
      Stub.ObjectFileFormat = !Stub.ObjectFileFormat.empty()
                                  ? Stub.ObjectFileFormat
                                  : TargetStub->ObjectFileFormat;

      if (Stub.IfsVersion != TargetStub->IfsVersion) {
        if (Stub.IfsVersion.getMajor() != IFSVersionCurrent.getMajor()) {
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
      if (Stub.ObjectFileFormat != TargetStub->ObjectFileFormat &&
          !TargetStub->ObjectFileFormat.empty()) {
        WithColor::error() << "Interface Stub: ObjectFileFormat Mismatch."
                           << "\nFilenames: " << PreviousInputFilePath << " "
                           << InputFilePath << "\nObjectFileFormat Values: "
                           << Stub.ObjectFileFormat << " "
                           << TargetStub->ObjectFileFormat << "\n";
        return -1;
      }
      if (Stub.Triple != TargetStub->Triple && !TargetStub->Triple.empty()) {
        WithColor::error() << "Interface Stub: Triple Mismatch."
                           << "\nFilenames: " << PreviousInputFilePath << " "
                           << InputFilePath
                           << "\nTriple Values: " << Stub.Triple << " "
                           << TargetStub->Triple << "\n";
        return -1;
      }
      if (Stub.SOName != TargetStub->SOName) {
        WithColor::error() << "Interface Stub: SOName Mismatch."
                           << "\nFilenames: " << PreviousInputFilePath << " "
                           << InputFilePath
                           << "\nSOName Values: " << Stub.SOName << " "
                           << TargetStub->SOName << "\n";
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

  if (Stub.IfsVersion != IFSVersionCurrent)
    if (Stub.IfsVersion.getMajor() != IFSVersionCurrent.getMajor()) {
      WithColor::error() << "Interface Stub: Bad IfsVersion: "
                         << Stub.IfsVersion << ", llvm-ifs supported version: "
                         << IFSVersionCurrent << ".\n";
      return -1;
    }

  for (auto &Entry : SymbolMap)
    Stub.Symbols.push_back(Entry.second);

  std::error_code SysErr;

  // Open file for writing.
  raw_fd_ostream Out(OutputFilename, SysErr);
  if (SysErr) {
    WithColor::error() << "Couldn't open " << OutputFilename
                       << " for writing.\n";
    return -1;
  }

  return writeIfso(Stub, (Action == "write-ifs"), Out);
}
