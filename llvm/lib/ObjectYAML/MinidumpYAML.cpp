//===- MinidumpYAML.cpp - Minidump YAMLIO implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/MinidumpYAML.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ConvertUTF.h"

using namespace llvm;
using namespace llvm::MinidumpYAML;
using namespace llvm::minidump;

namespace {
/// A helper class to manage the placement of various structures into the final
/// minidump binary. Space for objects can be allocated via various allocate***
/// methods, while the final minidump file is written by calling the writeTo
/// method. The plain versions of allocation functions take a reference to the
/// data which is to be written (and hence the data must be available until
/// writeTo is called), while the "New" versions allocate the data in an
/// allocator-managed buffer, which is available until the allocator object is
/// destroyed. For both kinds of functions, it is possible to modify the
/// data for which the space has been "allocated" until the final writeTo call.
/// This is useful for "linking" the allocated structures via their offsets.
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

  size_t allocateBytes(yaml::BinaryRef Data) {
    return allocateCallback(Data.binary_size(), [Data](raw_ostream &OS) {
      Data.writeAsBinary(OS);
    });
  }

  template <typename T> size_t allocateArray(ArrayRef<T> Data) {
    return allocateBytes({reinterpret_cast<const uint8_t *>(Data.data()),
                          sizeof(T) * Data.size()});
  }

  template <typename T, typename RangeType>
  std::pair<size_t, MutableArrayRef<T>>
  allocateNewArray(const iterator_range<RangeType> &Range);

  template <typename T> size_t allocateObject(const T &Data) {
    return allocateArray(makeArrayRef(Data));
  }

  template <typename T, typename... Types>
  std::pair<size_t, T *> allocateNewObject(Types &&... Args) {
    T *Object = new (Temporaries.Allocate<T>()) T(std::forward<Types>(Args)...);
    return {allocateObject(*Object), Object};
  }

  size_t allocateString(StringRef Str);

  void writeTo(raw_ostream &OS) const;

private:
  size_t NextOffset = 0;

  BumpPtrAllocator Temporaries;
  std::vector<std::function<void(raw_ostream &)>> Callbacks;
};
} // namespace

template <typename T, typename RangeType>
std::pair<size_t, MutableArrayRef<T>>
BlobAllocator::allocateNewArray(const iterator_range<RangeType> &Range) {
  size_t Num = std::distance(Range.begin(), Range.end());
  MutableArrayRef<T> Array(Temporaries.Allocate<T>(Num), Num);
  std::uninitialized_copy(Range.begin(), Range.end(), Array.begin());
  return {allocateArray(Array), Array};
}

size_t BlobAllocator::allocateString(StringRef Str) {
  SmallVector<UTF16, 32> WStr;
  bool OK = convertUTF8ToUTF16String(Str, WStr);
  assert(OK && "Invalid UTF8 in Str?");
  (void)OK;

  // The utf16 string is null-terminated, but the terminator is not counted in
  // the string size.
  WStr.push_back(0);
  size_t Result =
      allocateNewObject<support::ulittle32_t>(2 * (WStr.size() - 1)).first;
  allocateNewArray<support::ulittle16_t>(make_range(WStr.begin(), WStr.end()));
  return Result;
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
  case StreamType::MemoryList:
    return StreamKind::MemoryList;
  case StreamType::ModuleList:
    return StreamKind::ModuleList;
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
  case StreamType::ThreadList:
    return StreamKind::ThreadList;
  default:
    return StreamKind::RawContent;
  }
}

std::unique_ptr<Stream> Stream::create(StreamType Type) {
  StreamKind Kind = getKind(Type);
  switch (Kind) {
  case StreamKind::MemoryList:
    return llvm::make_unique<MemoryListStream>();
  case StreamKind::ModuleList:
    return llvm::make_unique<ModuleListStream>();
  case StreamKind::RawContent:
    return llvm::make_unique<RawContentStream>(Type);
  case StreamKind::SystemInfo:
    return llvm::make_unique<SystemInfoStream>();
  case StreamKind::TextContent:
    return llvm::make_unique<TextContentStream>(Type);
  case StreamKind::ThreadList:
    return llvm::make_unique<ThreadListStream>();
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

void yaml::MappingTraits<VSFixedFileInfo>::mapping(IO &IO,
                                                   VSFixedFileInfo &Info) {
  mapOptionalHex(IO, "Signature", Info.Signature, 0);
  mapOptionalHex(IO, "Struct Version", Info.StructVersion, 0);
  mapOptionalHex(IO, "File Version High", Info.FileVersionHigh, 0);
  mapOptionalHex(IO, "File Version Low", Info.FileVersionLow, 0);
  mapOptionalHex(IO, "Product Version High", Info.ProductVersionHigh, 0);
  mapOptionalHex(IO, "Product Version Low", Info.ProductVersionLow, 0);
  mapOptionalHex(IO, "File Flags Mask", Info.FileFlagsMask, 0);
  mapOptionalHex(IO, "File Flags", Info.FileFlags, 0);
  mapOptionalHex(IO, "File OS", Info.FileOS, 0);
  mapOptionalHex(IO, "File Type", Info.FileType, 0);
  mapOptionalHex(IO, "File Subtype", Info.FileSubtype, 0);
  mapOptionalHex(IO, "File Date High", Info.FileDateHigh, 0);
  mapOptionalHex(IO, "File Date Low", Info.FileDateLow, 0);
}

void yaml::MappingTraits<ModuleListStream::entry_type>::mapping(
    IO &IO, ModuleListStream::entry_type &M) {
  mapRequiredHex(IO, "Base of Image", M.Entry.BaseOfImage);
  mapRequiredHex(IO, "Size of Image", M.Entry.SizeOfImage);
  mapOptionalHex(IO, "Checksum", M.Entry.Checksum, 0);
  IO.mapOptional("Time Date Stamp", M.Entry.TimeDateStamp,
                 support::ulittle32_t(0));
  IO.mapRequired("Module Name", M.Name);
  IO.mapOptional("Version Info", M.Entry.VersionInfo, VSFixedFileInfo());
  IO.mapRequired("CodeView Record", M.CvRecord);
  IO.mapOptional("Misc Record", M.MiscRecord, yaml::BinaryRef());
  mapOptionalHex(IO, "Reserved0", M.Entry.Reserved0, 0);
  mapOptionalHex(IO, "Reserved1", M.Entry.Reserved1, 0);
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

void yaml::MappingTraits<MemoryListStream::entry_type>::mapping(
    IO &IO, MemoryListStream::entry_type &Range) {
  MappingContextTraits<MemoryDescriptor, yaml::BinaryRef>::mapping(
      IO, Range.Entry, Range.Content);
}

static void streamMapping(yaml::IO &IO, MemoryListStream &Stream) {
  IO.mapRequired("Memory Ranges", Stream.Entries);
}

static void streamMapping(yaml::IO &IO, ModuleListStream &Stream) {
  IO.mapRequired("Modules", Stream.Entries);
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

void yaml::MappingContextTraits<MemoryDescriptor, yaml::BinaryRef>::mapping(
    IO &IO, MemoryDescriptor &Memory, BinaryRef &Content) {
  mapRequiredHex(IO, "Start of Memory Range", Memory.StartOfMemoryRange);
  IO.mapRequired("Content", Content);
}

void yaml::MappingTraits<ThreadListStream::entry_type>::mapping(
    IO &IO, ThreadListStream::entry_type &T) {
  mapRequiredHex(IO, "Thread Id", T.Entry.ThreadId);
  mapOptionalHex(IO, "Suspend Count", T.Entry.SuspendCount, 0);
  mapOptionalHex(IO, "Priority Class", T.Entry.PriorityClass, 0);
  mapOptionalHex(IO, "Priority", T.Entry.Priority, 0);
  mapOptionalHex(IO, "Environment Block", T.Entry.EnvironmentBlock, 0);
  IO.mapRequired("Context", T.Context);
  IO.mapRequired("Stack", T.Entry.Stack, T.Stack);
}

static void streamMapping(yaml::IO &IO, ThreadListStream &Stream) {
  IO.mapRequired("Threads", Stream.Entries);
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
  case MinidumpYAML::Stream::StreamKind::MemoryList:
    streamMapping(IO, llvm::cast<MemoryListStream>(*S));
    break;
  case MinidumpYAML::Stream::StreamKind::ModuleList:
    streamMapping(IO, llvm::cast<ModuleListStream>(*S));
    break;
  case MinidumpYAML::Stream::StreamKind::RawContent:
    streamMapping(IO, llvm::cast<RawContentStream>(*S));
    break;
  case MinidumpYAML::Stream::StreamKind::SystemInfo:
    streamMapping(IO, llvm::cast<SystemInfoStream>(*S));
    break;
  case MinidumpYAML::Stream::StreamKind::TextContent:
    streamMapping(IO, llvm::cast<TextContentStream>(*S));
    break;
  case MinidumpYAML::Stream::StreamKind::ThreadList:
    streamMapping(IO, llvm::cast<ThreadListStream>(*S));
    break;
  }
}

StringRef yaml::MappingTraits<std::unique_ptr<Stream>>::validate(
    yaml::IO &IO, std::unique_ptr<MinidumpYAML::Stream> &S) {
  switch (S->Kind) {
  case MinidumpYAML::Stream::StreamKind::RawContent:
    return streamValidate(cast<RawContentStream>(*S));
  case MinidumpYAML::Stream::StreamKind::MemoryList:
  case MinidumpYAML::Stream::StreamKind::ModuleList:
  case MinidumpYAML::Stream::StreamKind::SystemInfo:
  case MinidumpYAML::Stream::StreamKind::TextContent:
  case MinidumpYAML::Stream::StreamKind::ThreadList:
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

static LocationDescriptor layout(BlobAllocator &File, yaml::BinaryRef Data) {
  return {support::ulittle32_t(Data.binary_size()),
          support::ulittle32_t(File.allocateBytes(Data))};
}

static void layout(BlobAllocator &File, MemoryListStream::entry_type &Range) {
  Range.Entry.Memory = layout(File, Range.Content);
}

static void layout(BlobAllocator &File, ModuleListStream::entry_type &M) {
  M.Entry.ModuleNameRVA = File.allocateString(M.Name);

  M.Entry.CvRecord = layout(File, M.CvRecord);
  M.Entry.MiscRecord = layout(File, M.MiscRecord);
}

static void layout(BlobAllocator &File, ThreadListStream::entry_type &T) {
  T.Entry.Stack.Memory = layout(File, T.Stack);
  T.Entry.Context = layout(File, T.Context);
}

template <typename EntryT>
static size_t layout(BlobAllocator &File,
                     MinidumpYAML::detail::ListStream<EntryT> &S) {

  File.allocateNewObject<support::ulittle32_t>(S.Entries.size());
  for (auto &E : S.Entries)
    File.allocateObject(E.Entry);

  size_t DataEnd = File.tell();

  // Lay out the auxiliary data, (which is not a part of the stream).
  DataEnd = File.tell();
  for (auto &E : S.Entries)
    layout(File, E);

  return DataEnd;
}

static Directory layout(BlobAllocator &File, Stream &S) {
  Directory Result;
  Result.Type = S.Type;
  Result.Location.RVA = File.tell();
  Optional<size_t> DataEnd;
  switch (S.Kind) {
  case Stream::StreamKind::MemoryList:
    DataEnd = layout(File, cast<MemoryListStream>(S));
    break;
  case Stream::StreamKind::ModuleList:
    DataEnd = layout(File, cast<ModuleListStream>(S));
    break;
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
  case Stream::StreamKind::ThreadList:
    DataEnd = layout(File, cast<ThreadListStream>(S));
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
  case StreamKind::MemoryList: {
    auto ExpectedList = File.getMemoryList();
    if (!ExpectedList)
      return ExpectedList.takeError();
    std::vector<MemoryListStream::entry_type> Ranges;
    for (const MemoryDescriptor &MD : *ExpectedList) {
      auto ExpectedContent = File.getRawData(MD.Memory);
      if (!ExpectedContent)
        return ExpectedContent.takeError();
      Ranges.push_back({MD, *ExpectedContent});
    }
    return llvm::make_unique<MemoryListStream>(std::move(Ranges));
  }
  case StreamKind::ModuleList: {
    auto ExpectedList = File.getModuleList();
    if (!ExpectedList)
      return ExpectedList.takeError();
    std::vector<ModuleListStream::entry_type> Modules;
    for (const Module &M : *ExpectedList) {
      auto ExpectedName = File.getString(M.ModuleNameRVA);
      if (!ExpectedName)
        return ExpectedName.takeError();
      auto ExpectedCv = File.getRawData(M.CvRecord);
      if (!ExpectedCv)
        return ExpectedCv.takeError();
      auto ExpectedMisc = File.getRawData(M.MiscRecord);
      if (!ExpectedMisc)
        return ExpectedMisc.takeError();
      Modules.push_back(
          {M, std::move(*ExpectedName), *ExpectedCv, *ExpectedMisc});
    }
    return llvm::make_unique<ModuleListStream>(std::move(Modules));
  }
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
  case StreamKind::ThreadList: {
    auto ExpectedList = File.getThreadList();
    if (!ExpectedList)
      return ExpectedList.takeError();
    std::vector<ThreadListStream::entry_type> Threads;
    for (const Thread &T : *ExpectedList) {
      auto ExpectedStack = File.getRawData(T.Stack.Memory);
      if (!ExpectedStack)
        return ExpectedStack.takeError();
      auto ExpectedContext = File.getRawData(T.Context);
      if (!ExpectedContext)
        return ExpectedContext.takeError();
      Threads.push_back({T, *ExpectedStack, *ExpectedContext});
    }
    return llvm::make_unique<ThreadListStream>(std::move(Threads));
  }
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
