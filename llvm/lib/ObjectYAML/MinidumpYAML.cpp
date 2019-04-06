//===- MinidumpYAML.cpp - Minidump YAMLIO implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/MinidumpYAML.h"
#include "llvm/Support/ConvertUTF.h"

using namespace llvm;
using namespace llvm::MinidumpYAML;
using namespace llvm::minidump;

namespace {
class BlobAllocator {
public:
  size_t tell() const { return NextOffset; }

  size_t allocateCallback(size_t Size,
                          std::function<void(raw_ostream &)> Callback) {
    size_t Offset = NextOffset;
    NextOffset += Size;
    Callbacks.push_back(std::move(Callback));
    return Offset;
  }

  size_t allocateBytes(ArrayRef<uint8_t> Data) {
    return allocateCallback(
        Data.size(), [Data](raw_ostream &OS) { OS << toStringRef(Data); });
  }

  template <typename T> size_t allocateArray(ArrayRef<T> Data) {
    return allocateBytes({reinterpret_cast<const uint8_t *>(Data.data()),
                          sizeof(T) * Data.size()});
  }

  template <typename T> size_t allocateObject(const T &Data) {
    return allocateArray(makeArrayRef(Data));
  }

  size_t allocateString(StringRef Str);

  void writeTo(raw_ostream &OS) const;

private:
  size_t NextOffset = 0;

  std::vector<std::function<void(raw_ostream &)>> Callbacks;
};
} // namespace

size_t BlobAllocator::allocateString(StringRef Str) {
  SmallVector<UTF16, 32> WStr;
  bool OK = convertUTF8ToUTF16String(Str, WStr);
  assert(OK && "Invalid UTF8 in Str?");
  (void)OK;

  SmallVector<support::ulittle16_t, 32> EndianStr(WStr.size() + 1,
                                                  support::ulittle16_t());
  copy(WStr, EndianStr.begin());
  return allocateCallback(
      sizeof(uint32_t) + EndianStr.size() * sizeof(support::ulittle16_t),
      [EndianStr](raw_ostream &OS) {
        // Length does not include the null-terminator.
        support::ulittle32_t Length(2 * (EndianStr.size() - 1));
        OS.write(reinterpret_cast<const char *>(&Length), sizeof(Length));
        OS.write(reinterpret_cast<const char *>(EndianStr.begin()),
                 sizeof(support::ulittle16_t) * EndianStr.size());
      });
}

void BlobAllocator::writeTo(raw_ostream &OS) const {
  size_t BeginOffset = OS.tell();
  for (const auto &Callback : Callbacks)
    Callback(OS);
  assert(OS.tell() == BeginOffset + NextOffset &&
         "Callbacks wrote an unexpected number of bytes.");
  (void)BeginOffset;
}

/// Perform an optional yaml-mapping of an endian-aware type EndianType. The
/// only purpose of this function is to avoid casting the Default value to the
/// endian type;
template <typename EndianType>
static inline void mapOptional(yaml::IO &IO, const char *Key, EndianType &Val,
                               typename EndianType::value_type Default) {
  IO.mapOptional(Key, Val, EndianType(Default));
}

/// Yaml-map an endian-aware type EndianType as some other type MapType.
template <typename MapType, typename EndianType>
static inline void mapRequiredAs(yaml::IO &IO, const char *Key,
                                 EndianType &Val) {
  MapType Mapped = static_cast<typename EndianType::value_type>(Val);
  IO.mapRequired(Key, Mapped);
  Val = static_cast<typename EndianType::value_type>(Mapped);
}

/// Perform an optional yaml-mapping of an endian-aware type EndianType as some
/// other type MapType.
template <typename MapType, typename EndianType>
static inline void mapOptionalAs(yaml::IO &IO, const char *Key, EndianType &Val,
                                 MapType Default) {
  MapType Mapped = static_cast<typename EndianType::value_type>(Val);
  IO.mapOptional(Key, Mapped, Default);
  Val = static_cast<typename EndianType::value_type>(Mapped);
}

namespace {
/// Return the appropriate yaml Hex type for a given endian-aware type.
template <typename EndianType> struct HexType;
template <> struct HexType<support::ulittle16_t> { using type = yaml::Hex16; };
template <> struct HexType<support::ulittle32_t> { using type = yaml::Hex32; };
template <> struct HexType<support::ulittle64_t> { using type = yaml::Hex64; };
} // namespace

/// Yaml-map an endian-aware type as an appropriately-sized hex value.
template <typename EndianType>
static inline void mapRequiredHex(yaml::IO &IO, const char *Key,
                                  EndianType &Val) {
  mapRequiredAs<typename HexType<EndianType>::type>(IO, Key, Val);
}

/// Perform an optional yaml-mapping of an endian-aware type as an
/// appropriately-sized hex value.
template <typename EndianType>
static inline void mapOptionalHex(yaml::IO &IO, const char *Key,
                                  EndianType &Val,
                                  typename EndianType::value_type Default) {
  mapOptionalAs<typename HexType<EndianType>::type>(IO, Key, Val, Default);
}

Stream::~Stream() = default;

Stream::StreamKind Stream::getKind(StreamType Type) {
  switch (Type) {
  case StreamType::SystemInfo:
    return StreamKind::SystemInfo;
  case StreamType::LinuxCPUInfo:
  case StreamType::LinuxProcStatus:
  case StreamType::LinuxLSBRelease:
  case StreamType::LinuxCMDLine:
  case StreamType::LinuxMaps:
  case StreamType::LinuxProcStat:
  case StreamType::LinuxProcUptime:
    return StreamKind::TextContent;
  default:
    return StreamKind::RawContent;
  }
}

std::unique_ptr<Stream> Stream::create(StreamType Type) {
  StreamKind Kind = getKind(Type);
  switch (Kind) {
  case StreamKind::RawContent:
    return llvm::make_unique<RawContentStream>(Type);
  case StreamKind::SystemInfo:
    return llvm::make_unique<SystemInfoStream>();
  case StreamKind::TextContent:
    return llvm::make_unique<TextContentStream>(Type);
  }
  llvm_unreachable("Unhandled stream kind!");
}

void yaml::ScalarEnumerationTraits<ProcessorArchitecture>::enumeration(
    IO &IO, ProcessorArchitecture &Arch) {
#define HANDLE_MDMP_ARCH(CODE, NAME)                                           \
  IO.enumCase(Arch, #NAME, ProcessorArchitecture::NAME);
#include "llvm/BinaryFormat/MinidumpConstants.def"
  IO.enumFallback<Hex16>(Arch);
}

void yaml::ScalarEnumerationTraits<OSPlatform>::enumeration(IO &IO,
                                                            OSPlatform &Plat) {
#define HANDLE_MDMP_PLATFORM(CODE, NAME)                                       \
  IO.enumCase(Plat, #NAME, OSPlatform::NAME);
#include "llvm/BinaryFormat/MinidumpConstants.def"
  IO.enumFallback<Hex32>(Plat);
}

void yaml::ScalarEnumerationTraits<StreamType>::enumeration(IO &IO,
                                                            StreamType &Type) {
#define HANDLE_MDMP_STREAM_TYPE(CODE, NAME)                                    \
  IO.enumCase(Type, #NAME, StreamType::NAME);
#include "llvm/BinaryFormat/MinidumpConstants.def"
  IO.enumFallback<Hex32>(Type);
}

void yaml::MappingTraits<CPUInfo::ArmInfo>::mapping(IO &IO,
                                                    CPUInfo::ArmInfo &Info) {
  mapRequiredHex(IO, "CPUID", Info.CPUID);
  mapOptionalHex(IO, "ELF hwcaps", Info.ElfHWCaps, 0);
}

namespace {
template <std::size_t N> struct FixedSizeHex {
  FixedSizeHex(uint8_t (&Storage)[N]) : Storage(Storage) {}

  uint8_t (&Storage)[N];
};
} // namespace

namespace llvm {
namespace yaml {
template <std::size_t N> struct ScalarTraits<FixedSizeHex<N>> {
  static void output(const FixedSizeHex<N> &Fixed, void *, raw_ostream &OS) {
    OS << toHex(makeArrayRef(Fixed.Storage));
  }

  static StringRef input(StringRef Scalar, void *, FixedSizeHex<N> &Fixed) {
    if (!all_of(Scalar, isHexDigit))
      return "Invalid hex digit in input";
    if (Scalar.size() < 2 * N)
      return "String too short";
    if (Scalar.size() > 2 * N)
      return "String too long";
    copy(fromHex(Scalar), Fixed.Storage);
    return "";
  }

  static QuotingType mustQuote(StringRef S) { return QuotingType::None; }
};
} // namespace yaml
} // namespace llvm
void yaml::MappingTraits<CPUInfo::OtherInfo>::mapping(
    IO &IO, CPUInfo::OtherInfo &Info) {
  FixedSizeHex<sizeof(Info.ProcessorFeatures)> Features(Info.ProcessorFeatures);
  IO.mapRequired("Features", Features);
}

namespace {
/// A type which only accepts strings of a fixed size for yaml conversion.
template <std::size_t N> struct FixedSizeString {
  FixedSizeString(char (&Storage)[N]) : Storage(Storage) {}

  char (&Storage)[N];
};
} // namespace

namespace llvm {
namespace yaml {
template <std::size_t N> struct ScalarTraits<FixedSizeString<N>> {
  static void output(const FixedSizeString<N> &Fixed, void *, raw_ostream &OS) {
    OS << StringRef(Fixed.Storage, N);
  }

  static StringRef input(StringRef Scalar, void *, FixedSizeString<N> &Fixed) {
    if (Scalar.size() < N)
      return "String too short";
    if (Scalar.size() > N)
      return "String too long";
    copy(Scalar, Fixed.Storage);
    return "";
  }

  static QuotingType mustQuote(StringRef S) { return needsQuotes(S); }
};
} // namespace yaml
} // namespace llvm

void yaml::MappingTraits<CPUInfo::X86Info>::mapping(IO &IO,
                                                    CPUInfo::X86Info &Info) {
  FixedSizeString<sizeof(Info.VendorID)> VendorID(Info.VendorID);
  IO.mapRequired("Vendor ID", VendorID);

  mapRequiredHex(IO, "Version Info", Info.VersionInfo);
  mapRequiredHex(IO, "Feature Info", Info.FeatureInfo);
  mapOptionalHex(IO, "AMD Extended Features", Info.AMDExtendedFeatures, 0);
}

static void streamMapping(yaml::IO &IO, RawContentStream &Stream) {
  IO.mapOptional("Content", Stream.Content);
  IO.mapOptional("Size", Stream.Size, Stream.Content.binary_size());
}

static StringRef streamValidate(RawContentStream &Stream) {
  if (Stream.Size.value < Stream.Content.binary_size())
    return "Stream size must be greater or equal to the content size";
  return "";
}

static void streamMapping(yaml::IO &IO, SystemInfoStream &Stream) {
  SystemInfo &Info = Stream.Info;
  IO.mapRequired("Processor Arch", Info.ProcessorArch);
  mapOptional(IO, "Processor Level", Info.ProcessorLevel, 0);
  mapOptional(IO, "Processor Revision", Info.ProcessorRevision, 0);
  IO.mapOptional("Number of Processors", Info.NumberOfProcessors, 0);
  IO.mapOptional("Product type", Info.ProductType, 0);
  mapOptional(IO, "Major Version", Info.MajorVersion, 0);
  mapOptional(IO, "Minor Version", Info.MinorVersion, 0);
  mapOptional(IO, "Build Number", Info.BuildNumber, 0);
  IO.mapRequired("Platform ID", Info.PlatformId);
  IO.mapOptional("CSD Version", Stream.CSDVersion, "");
  mapOptionalHex(IO, "Suite Mask", Info.SuiteMask, 0);
  mapOptionalHex(IO, "Reserved", Info.Reserved, 0);
  switch (static_cast<ProcessorArchitecture>(Info.ProcessorArch)) {
  case ProcessorArchitecture::X86:
  case ProcessorArchitecture::AMD64:
    IO.mapOptional("CPU", Info.CPU.X86);
    break;
  case ProcessorArchitecture::ARM:
  case ProcessorArchitecture::ARM64:
    IO.mapOptional("CPU", Info.CPU.Arm);
    break;
  default:
    IO.mapOptional("CPU", Info.CPU.Other);
    break;
  }
}

static void streamMapping(yaml::IO &IO, TextContentStream &Stream) {
  IO.mapOptional("Text", Stream.Text);
}

void yaml::MappingTraits<std::unique_ptr<Stream>>::mapping(
    yaml::IO &IO, std::unique_ptr<MinidumpYAML::Stream> &S) {
  StreamType Type;
  if (IO.outputting())
    Type = S->Type;
  IO.mapRequired("Type", Type);

  if (!IO.outputting())
    S = MinidumpYAML::Stream::create(Type);
  switch (S->Kind) {
  case MinidumpYAML::Stream::StreamKind::RawContent:
    streamMapping(IO, llvm::cast<RawContentStream>(*S));
    break;
  case MinidumpYAML::Stream::StreamKind::SystemInfo:
    streamMapping(IO, llvm::cast<SystemInfoStream>(*S));
    break;
  case MinidumpYAML::Stream::StreamKind::TextContent:
    streamMapping(IO, llvm::cast<TextContentStream>(*S));
    break;
  }
}

StringRef yaml::MappingTraits<std::unique_ptr<Stream>>::validate(
    yaml::IO &IO, std::unique_ptr<MinidumpYAML::Stream> &S) {
  switch (S->Kind) {
  case MinidumpYAML::Stream::StreamKind::RawContent:
    return streamValidate(cast<RawContentStream>(*S));
  case MinidumpYAML::Stream::StreamKind::SystemInfo:
  case MinidumpYAML::Stream::StreamKind::TextContent:
    return "";
  }
  llvm_unreachable("Fully covered switch above!");
}

void yaml::MappingTraits<Object>::mapping(IO &IO, Object &O) {
  IO.mapTag("!minidump", true);
  mapOptionalHex(IO, "Signature", O.Header.Signature, Header::MagicSignature);
  mapOptionalHex(IO, "Version", O.Header.Version, Header::MagicVersion);
  mapOptionalHex(IO, "Flags", O.Header.Flags, 0);
  IO.mapRequired("Streams", O.Streams);
}

static Directory layout(BlobAllocator &File, Stream &S) {
  Directory Result;
  Result.Type = S.Type;
  Result.Location.RVA = File.tell();
  Optional<size_t> DataEnd;
  switch (S.Kind) {
  case Stream::StreamKind::RawContent: {
    RawContentStream &Raw = cast<RawContentStream>(S);
    File.allocateCallback(Raw.Size, [&Raw](raw_ostream &OS) {
      Raw.Content.writeAsBinary(OS);
      assert(Raw.Content.binary_size() <= Raw.Size);
      OS << std::string(Raw.Size - Raw.Content.binary_size(), '\0');
    });
    break;
  }
  case Stream::StreamKind::SystemInfo: {
    SystemInfoStream &SystemInfo = cast<SystemInfoStream>(S);
    File.allocateObject(SystemInfo.Info);
    // The CSD string is not a part of the stream.
    DataEnd = File.tell();
    SystemInfo.Info.CSDVersionRVA = File.allocateString(SystemInfo.CSDVersion);
    break;
  }
  case Stream::StreamKind::TextContent:
    File.allocateArray(arrayRefFromStringRef(cast<TextContentStream>(S).Text));
    break;
  }
  // If DataEnd is not set, we assume everything we generated is a part of the
  // stream.
  Result.Location.DataSize =
      DataEnd.getValueOr(File.tell()) - Result.Location.RVA;
  return Result;
}

void MinidumpYAML::writeAsBinary(Object &Obj, raw_ostream &OS) {
  BlobAllocator File;
  File.allocateObject(Obj.Header);

  std::vector<Directory> StreamDirectory(Obj.Streams.size());
  Obj.Header.StreamDirectoryRVA =
      File.allocateArray(makeArrayRef(StreamDirectory));
  Obj.Header.NumberOfStreams = StreamDirectory.size();

  for (auto &Stream : enumerate(Obj.Streams))
    StreamDirectory[Stream.index()] = layout(File, *Stream.value());

  File.writeTo(OS);
}

Error MinidumpYAML::writeAsBinary(StringRef Yaml, raw_ostream &OS) {
  yaml::Input Input(Yaml);
  Object Obj;
  Input >> Obj;
  if (std::error_code EC = Input.error())
    return errorCodeToError(EC);

  writeAsBinary(Obj, OS);
  return Error::success();
}

Expected<std::unique_ptr<Stream>>
Stream::create(const Directory &StreamDesc, const object::MinidumpFile &File) {
  StreamKind Kind = getKind(StreamDesc.Type);
  switch (Kind) {
  case StreamKind::RawContent:
    return llvm::make_unique<RawContentStream>(StreamDesc.Type,
                                               File.getRawStream(StreamDesc));
  case StreamKind::SystemInfo: {
    auto ExpectedInfo = File.getSystemInfo();
    if (!ExpectedInfo)
      return ExpectedInfo.takeError();
    auto ExpectedCSDVersion = File.getString(ExpectedInfo->CSDVersionRVA);
    if (!ExpectedCSDVersion)
      return ExpectedInfo.takeError();
    return llvm::make_unique<SystemInfoStream>(*ExpectedInfo,
                                               std::move(*ExpectedCSDVersion));
  }
  case StreamKind::TextContent:
    return llvm::make_unique<TextContentStream>(
        StreamDesc.Type, toStringRef(File.getRawStream(StreamDesc)));
  }
  llvm_unreachable("Unhandled stream kind!");
}

Expected<Object> Object::create(const object::MinidumpFile &File) {
  std::vector<std::unique_ptr<Stream>> Streams;
  Streams.reserve(File.streams().size());
  for (const Directory &StreamDesc : File.streams()) {
    auto ExpectedStream = Stream::create(StreamDesc, File);
    if (!ExpectedStream)
      return ExpectedStream.takeError();
    Streams.push_back(std::move(*ExpectedStream));
  }
  return Object(File.header(), std::move(Streams));
}
