//===- DXContainer.cpp - DXContainer object file implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/DXContainer.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/SubtargetFeature.h"

using namespace llvm;
using namespace llvm::object;

static Error parseFailed(const Twine &Msg) {
  return make_error<GenericBinaryError>(Msg.str(), object_error::parse_failed);
}

static bool readIsOutOfBounds(StringRef Buffer, const char *Src, size_t Size) {
  return !Src || Size > static_cast<size_t>(Buffer.end() - Src);
}

template <typename T, bool FixEndianness = true>
static Error readStruct(StringRef Buffer, const char *Src, T &Struct) {
  // Don't read before the beginning or past the end of the file
  if (readIsOutOfBounds(Buffer, Src, sizeof(T)))
    return parseFailed("Reading structure out of file bounds");

  memcpy(&Struct, Src, sizeof(T));
  // DXContainer is always little endian
  if constexpr (FixEndianness)
    if (sys::IsBigEndianHost)
      Struct.swapBytes();
  return Error::success();
}

template <typename T>
static Error readInteger(StringRef Buffer, const char *Src, T &Val,
                         Twine Str = "structure") {
  static_assert(std::is_integral_v<T>,
                "Cannot call readInteger on non-integral type.");
  // Don't read before the beginning or past the end of the file
  if (readIsOutOfBounds(Buffer, Src, sizeof(T)))
    return parseFailed(Twine("Reading ") + Str + " out of file bounds");

  // The DXContainer offset table is comprised of uint32_t values but not padded
  // to a 64-bit boundary. So Parts may start unaligned if there is an odd
  // number of parts and part data itself is not required to be padded.
  if (reinterpret_cast<uintptr_t>(Src) % alignof(T) != 0)
    memcpy(reinterpret_cast<char *>(&Val), Src, sizeof(T));
  else
    Val = *reinterpret_cast<const T *>(Src);
  // DXContainer is always little endian
  if (sys::IsBigEndianHost)
    sys::swapByteOrder(Val);
  return Error::success();
}

/// Read a null-terminated string at the position Src from Buffer, with maximum
/// byte size of MaxSize (including the null-terminator). Advance Src by the
/// number of bytes read.
static Error readString(StringRef Buffer, const char *&Src, size_t MaxSize,
                        StringRef &Val, Twine Desc) {
  if (readIsOutOfBounds(Buffer, Src, MaxSize))
    return parseFailed(Desc + " is out of file bounds");

  // Ensure that the null-terminator is somewhere within MaxSize bytes.
  Buffer = Buffer.substr(Src - Buffer.data(), MaxSize);
  size_t Length = Buffer.find('\0');
  if (Length == Buffer.npos)
    return parseFailed(Desc + " does not end with null-terminator");

  Val = StringRef(Buffer.data(), Length);
  Src += Length + 1;
  return Error::success();
}

DXContainer::DXContainer(MemoryBufferRef O) : Data(O) {}

Error DXContainer::parseHeader() {
  if (Error Err = readStruct(Data.getBuffer(), Data.getBuffer().data(), Header))
    return Err;
  if (StringRef(reinterpret_cast<char *>(Header.Magic), 4) != "DXBC")
    return parseFailed("Missing DXBC header magic");
  return Error::success();
}

Error DXContainer::parseDXILHeader(dxbc::PartType PT, StringRef Part) {
  bool IsDebug = dxbc::isDebugProgramPart(PT);
  std::optional<DXILData> &DXIL = IsDebug ? this->DebugDXIL : this->DXIL;

  if (DXIL)
    return parseFailed(formatv("more than one {0} part is present in the file",
                               dxbc::getProgramPartName(IsDebug)));
  const char *Current = Part.begin();
  dxbc::ProgramHeader Header;
  if (Error Err = readStruct(Part, Current, Header))
    return Err;
  Current += offsetof(dxbc::ProgramHeader, Bitcode) + Header.Bitcode.Offset;
  DXIL.emplace(std::make_pair(Header, Current));
  return Error::success();
}

Error DXContainer::parseDebugName(StringRef Part) {
  if (DebugName)
    return parseFailed("more than one ILDN part is present in the file");
  const char *Current = Part.begin();
  dxbc::DebugNameHeader Header;
  if (Error Err = readStruct(Part, Current, Header))
    return Err;
  Current += sizeof(Header);

  StringRef Name;
  if (Error Err = readString(Part, Current, Header.NameLength + 1, Name,
                             "debug file name"))
    return Err;
  if (Name.size() != Header.NameLength)
    return parseFailed("debug file name length mismatch");
  DebugName.emplace(Header, Name.data());

  return Error::success();
}

Error DXContainer::parsePrivateData(StringRef Part) {
  if (PrivateData)
    return parseFailed("more than one PRIV part is present in the file");
  PrivateData.emplace(Part);
  return Error::success();
}

Error DXContainer::parseShaderFeatureFlags(StringRef Part) {
  if (ShaderFeatureFlags)
    return parseFailed("More than one SFI0 part is present in the file");
  uint64_t FlagValue = 0;
  if (Error Err = readInteger(Part, Part.begin(), FlagValue))
    return Err;
  ShaderFeatureFlags = FlagValue;
  return Error::success();
}

Error DXContainer::parseHash(StringRef Part) {
  if (Hash)
    return parseFailed("More than one HASH part is present in the file");
  dxbc::ShaderHash ReadHash;
  if (Error Err = readStruct(Part, Part.begin(), ReadHash))
    return Err;
  Hash = ReadHash;
  return Error::success();
}

Error DXContainer::parseRootSignature(StringRef Part) {
  if (RootSignature)
    return parseFailed("More than one RTS0 part is present in the file");
  RootSignature = DirectX::RootSignature(Part);
  if (Error Err = RootSignature->parse())
    return Err;
  return Error::success();
}

Error DXContainer::parsePSVInfo(StringRef Part) {
  if (PSVInfo)
    return parseFailed("More than one PSV0 part is present in the file");
  PSVInfo = DirectX::PSVRuntimeInfo(Part);
  // Parsing the PSVRuntime info occurs late because we need to read data from
  // other parts first.
  return Error::success();
}

Error DirectX::Signature::initialize(StringRef Part) {
  dxbc::ProgramSignatureHeader SigHeader;
  if (Error Err = readStruct(Part, Part.begin(), SigHeader))
    return Err;
  size_t Size = sizeof(dxbc::ProgramSignatureElement) * SigHeader.ParamCount;

  if (Part.size() < Size + SigHeader.FirstParamOffset)
    return parseFailed("Signature parameters extend beyond the part boundary");

  Parameters.Data = Part.substr(SigHeader.FirstParamOffset, Size);

  StringTableOffset = SigHeader.FirstParamOffset + static_cast<uint32_t>(Size);
  StringTable = Part.substr(SigHeader.FirstParamOffset + Size);

  for (const auto &Param : Parameters) {
    if (Param.NameOffset < StringTableOffset)
      return parseFailed("Invalid parameter name offset: name starts before "
                         "the first name offset");
    if (Param.NameOffset - StringTableOffset > StringTable.size())
      return parseFailed("Invalid parameter name offset: name starts after the "
                         "end of the part data");
  }
  return Error::success();
}

Error DXContainer::parseCompilerVersionInfo(StringRef Part) {
  if (VersionInfo)
    return parseFailed("more than one VERS part is present in the file");
  const char *Current = Part.begin();
  dxbc::CompilerVersionHeader Header;
  if (Error Err = readStruct(Part, Current, Header))
    return Err;
  Current += sizeof(Header);

  if (!dxbc::isValidCompilerVersionFlags(to_underlying(Header.Flags)))
    return parseFailed("Incorrect shader compiler version flags combination");

  StringRef CommitSha;
  const char *Prev = Current;
  if (Error Err = readString(Part, Current, Header.ContentSizeInBytes,
                             CommitSha, "CommitSha"))
    return Err;
  StringRef CustomVersionString;
  if (Error Err = readString(Part, Current,
                             Header.ContentSizeInBytes - (Current - Prev),
                             CustomVersionString, "CustomVersionString"))
    return Err;

  VersionInfo.emplace();
  VersionInfo->Parameters = Header;
  VersionInfo->CommitSha = CommitSha;
  VersionInfo->CustomVersionString = CustomVersionString;
  return Error::success();
}

static Expected<size_t> parseNames(StringRef Section,
                                   mcdxbc::SourceInfo::SourceNames &Names) {
  const char *Current = Section.begin();
  dxbc::SourceInfo::Names::HeaderOnDisk HeaderOnDisk;
  if (Error Err = readStruct<decltype(HeaderOnDisk), false>(Section, Current,
                                                            HeaderOnDisk))
    return Err;
  Names.Parameters = HeaderOnDisk;
  Current += sizeof(HeaderOnDisk);

  if (Names.Parameters.Flags)
    return parseFailed("SRCI Names header flags must be zero");
  if (Current + Names.Parameters.EntriesSizeInBytes > Section.end())
    return parseFailed(
        "SRCI Names section content ends beyond the section boundary");

  Names.Entries.reserve(Names.Parameters.Count);
  for (size_t I : llvm::seq(Names.Parameters.Count)) {
    auto &Entry = Names.Entries.emplace_back();
    if (Error Err = readStruct(Section, Current, Entry.Parameters))
      return Err;

    const char *Next = Current + Entry.Parameters.AlignedSizeInBytes;
    if (Next > Section.end())
      return parseFailed(
          formatv("SRCI Names entry {0} ends beyond the section boundary", I));
    if (Entry.Parameters.Flags)
      return parseFailed(formatv("SRCI Names entry {0} flags must be zero", I));

    const char *FileName = Current + sizeof(Entry.Parameters);
    if (Error Err = readString(
            Section, FileName, Entry.Parameters.NameSizeInBytes, Entry.FileName,
            Twine("SRCI Names entry ") + Twine(I) + Twine(" file name")))
      return Err;
    if (FileName > Next)
      return parseFailed(formatv(
          "SRCI Names entry {0} file name ends beyond the entry boundary", I));
    Current = Next;
  }

  return Current - Section.begin();
}

static Expected<size_t>
parseUncompressedContentsEntries(StringRef Entries,
                                 mcdxbc::SourceInfo::SourceContents &Contents) {
  const char *Current = Entries.begin();

  Contents.Entries.reserve(Contents.Parameters.Count);
  for (size_t I : llvm::seq(Contents.Parameters.Count)) {
    auto &Entry = Contents.Entries.emplace_back();
    if (Error Err = readStruct(Entries, Current, Entry.Parameters))
      return Err;

    const char *Next = Current + Entry.Parameters.AlignedSizeInBytes;
    if (Next > Entries.end())
      return parseFailed(formatv(
          "SRCI Contents entry {0} ends beyond the section boundary", I));
    if (Entry.Parameters.Flags)
      return parseFailed(
          formatv("SRCI Contents entry {0} flags must be zero", I));

    const char *FileContentPtr = Current + sizeof(Entry.Parameters);
    const char *FileContentEndPtr = FileContentPtr;
    StringRef FileContent;
    if (Error Err = readString(Entries, FileContentEndPtr,
                               Entry.Parameters.ContentSizeInBytes, FileContent,
                               Twine("SRCI Contents entry ") + Twine(I) +
                                   Twine(" file content")))
      return Err;
    if (FileContentEndPtr - FileContentPtr !=
        Entry.Parameters.ContentSizeInBytes)
      return parseFailed(
          formatv("file size from header ({0} bytes) does not match content "
                  "size in SRCI Contents entry {1} ({2} bytes)",
                  FileContentEndPtr - FileContentPtr, I,
                  Entry.Parameters.ContentSizeInBytes));
    if (FileContentEndPtr > Next)
      return parseFailed(formatv(
          "SRCI Contents entry {0} file content ends beyond the entry boundary",
          I));

    Entry.FileContent = std::string(FileContent.data(),
                                    Entry.Parameters.ContentSizeInBytes - 1);

    Current = Next;
  }

  return Current - Entries.begin();
}

static Expected<size_t>
parseContentsEntries(StringRef Entries,
                     mcdxbc::SourceInfo::SourceContents &Contents) {
  using dxbc::SourceInfo::Contents::CompressionType;

  if (!dxbc::SourceInfo::Contents::isValidCompressionType(
          to_underlying(Contents.Parameters.Type)))
    return parseFailed("SRCI Contents section uses unknown compression type");

  SmallVector<uint8_t> UncompressedEntriesData;
  switch (Contents.Parameters.Type) {
  case CompressionType::None: {
    if (Contents.Parameters.EntriesSizeInBytes !=
        Contents.Parameters.UncompressedEntriesSizeInBytes)
      return parseFailed(formatv(
          "SRCI Contents is not compressed, but compressed size ({0} bytes) "
          "doesn't match uncompressed size ({1} bytes) in section header",
          Contents.Parameters.EntriesSizeInBytes,
          Contents.Parameters.UncompressedEntriesSizeInBytes));

    return parseUncompressedContentsEntries(Entries, Contents);
  }
  case CompressionType::Zlib: {
    if (!compression::zlib::isAvailable())
      return parseFailed(formatv(
          "SRCI Contents is compressed with Zlib, but {0}",
          compression::getReasonIfUnsupported(compression::Format::Zlib)));
    if (Error Err = compression::zlib::decompress(
            ArrayRef(reinterpret_cast<const uint8_t *>(Entries.begin()),
                     Contents.Parameters.EntriesSizeInBytes),
            UncompressedEntriesData,
            Contents.Parameters.UncompressedEntriesSizeInBytes))
      return Err;

    if (UncompressedEntriesData.size() !=
        Contents.Parameters.UncompressedEntriesSizeInBytes)
      return parseFailed("SRCI Contents uncompressed size from header does not "
                         "match with actual content size");

    if (Error Err = parseUncompressedContentsEntries(
                        StringRef(reinterpret_cast<const char *>(
                                      UncompressedEntriesData.data()),
                                  UncompressedEntriesData.size()),
                        Contents)
                        .takeError())
      return Err;

    return Contents.Parameters.EntriesSizeInBytes;
  }
  }
  llvm_unreachable("unhandled compression type");
}

static Expected<size_t>
parseContents(StringRef Section, mcdxbc::SourceInfo::SourceContents &Contents) {
  const char *Current = Section.begin();
  if (Error Err = readStruct(Section, Current, Contents.Parameters))
    return Err;
  size_t BytesRead = sizeof(Contents.Parameters);
  Current += BytesRead;

  if (Section.begin() + Contents.Parameters.EntriesSizeInBytes > Section.end())
    return parseFailed(
        formatv("SRCI Contents section ends beyond the section boundary"));
  if (Contents.Parameters.Flags)
    return parseFailed("SRCI Contents header flags must be zero");
  if (Current + Contents.Parameters.EntriesSizeInBytes > Section.end())
    return parseFailed(
        formatv("SRCI Contents entries end beyond the section boundary"));

  size_t BodyBytesRead = 0;
  if (Error Err = parseContentsEntries(Section.substr(BytesRead), Contents)
                      .moveInto(BodyBytesRead))
    return Err;
  return BytesRead + BodyBytesRead;
}

static Expected<size_t> parseArgs(StringRef Section,
                                  mcdxbc::SourceInfo::ProgramArgs &Args) {
  const char *Current = Section.begin();
  if (Error Err = readStruct(Section, Current, Args.Parameters))
    return Err;
  Current += sizeof(Args.Parameters);

  if (Args.Parameters.Flags)
    return parseFailed("SRCI Args header flags must be zero");
  if (Current + Args.Parameters.SizeInBytes > Section.end())
    return parseFailed(
        formatv("SRCI Args entries end beyond the section boundary", Section));

  Args.Args.reserve(Args.Parameters.Count);
  for (size_t I : llvm::seq(Args.Parameters.Count)) {
    auto &Entry = Args.Args.emplace_back();
    if (Error Err =
            readString(Section, Current, Section.end() - Current, Entry.first,
                       Twine("SRCI Args entry ") + Twine(I) + Twine(" name")))
      return Err;
    if (Error Err =
            readString(Section, Current, Section.end() - Current, Entry.second,
                       Twine("SRCI Args entry ") + Twine(I) + Twine(" value")))
      return Err;
  }

  return Current - Section.begin();
}

static Expected<size_t>
parseSourceInfoSection(const dxbc::SourceInfo::SectionHeader &Header,
                       StringRef SectionData, mcdxbc::SourceInfo &SourceInfo) {
  using dxbc::SourceInfo::SectionType;
  switch (Header.Type) {
  case SectionType::SourceNames: {
    SourceInfo.Names.GenericHeader = Header;
    return parseNames(SectionData, SourceInfo.Names);
  }
  case SectionType::SourceContents: {
    SourceInfo.Contents.GenericHeader = Header;
    return parseContents(SectionData, SourceInfo.Contents);
  }
  case SectionType::Args: {
    SourceInfo.Args.GenericHeader = Header;
    return parseArgs(SectionData, SourceInfo.Args);
  }
  }

  llvm_unreachable("Unknown source info section type");
}

Error DXContainer::parseSourceInfo(StringRef Part) {
  using dxbc::SourceInfo::SectionType;

  if (SourceInfo)
    return parseFailed("more than one SRCI part is present in the file");
  SourceInfo.emplace();

  const char *Current = Part.begin();
  if (Error Err = readStruct(Part, Current, SourceInfo->Parameters))
    return Err;
  Current += sizeof(SourceInfo->Parameters);

  if (SourceInfo->Parameters.AlignedSizeInBytes > Part.size())
    return parseFailed(formatv("size field in SRCI header ({0} bytes) is "
                               "greater than SRCI part size ({1} bytes)",
                               SourceInfo->Parameters.AlignedSizeInBytes,
                               Part.size()));
  if (SourceInfo->Parameters.Flags)
    return parseFailed("SRCI header flags must be zero");
  if (SourceInfo->Parameters.SectionCount != 3)
    return parseFailed("SRCI part must contain 3 sections");

  bool IsSectionPresent[to_underlying(
                            SectionType::LLVM_BITMASK_LARGEST_ENUMERATOR) +
                        1];
  std::fill(IsSectionPresent,
            IsSectionPresent +
                sizeof(IsSectionPresent) / sizeof(*IsSectionPresent),
            false);
  for (uint32_t Section = 0; Section < SourceInfo->Parameters.SectionCount;
       ++Section) {
    dxbc::SourceInfo::SectionHeader SectionHeader;
    if (Error Err = readStruct(Part, Current, SectionHeader))
      return Err;
    size_t BytesRead = sizeof(SectionHeader);

    StringRef SectionName =
        dxbc::SourceInfo::getSectionName(SectionHeader.Type);
    if (Current + SectionHeader.AlignedSizeInBytes > Part.end())
      return parseFailed(
          formatv("SRCI section {0} (#{1}) extends beyond the part boundary",
                  SectionName, Section));
    if (SectionHeader.Flags)
      return parseFailed(
          formatv("SRCI section {0} (#{1}) header flags must be zero",
                  SectionName, Section));

    size_t SectionTypeIdx = to_underlying(SectionHeader.Type);
    if (!dxbc::SourceInfo::isValidSectionType(SectionTypeIdx))
      return parseFailed(
          formatv("unknown SRCI section type {0}", SectionTypeIdx));
    if (IsSectionPresent[SectionTypeIdx])
      return parseFailed(formatv(
          "more than one {0} section is present in SRCI part", SectionName));
    IsSectionPresent[SectionTypeIdx] = true;

    size_t SectionBytesRead = 0;
    if (Error Err = parseSourceInfoSection(
                        SectionHeader,
                        Part.substr(Current + BytesRead - Part.begin(),
                                    SectionHeader.AlignedSizeInBytes),
                        *SourceInfo)
                        .moveInto(SectionBytesRead))
      return Err;
    BytesRead += SectionBytesRead;
    BytesRead = alignTo<dxbc::DXCONTAINER_STRUCT_ALIGNMENT>(BytesRead);

    if (BytesRead != SectionHeader.AlignedSizeInBytes)
      return parseFailed(formatv(
          "size of SRCI section {0} (#{1} - {2} bytes) does not match size "
          "specified in generic header ({3} bytes)",
          SectionName, Section, BytesRead, SectionHeader.AlignedSizeInBytes));
    Current += SectionHeader.AlignedSizeInBytes;
  }

  if (SourceInfo->Contents.Parameters.Count !=
      SourceInfo->Names.Parameters.Count)
    return parseFailed(
        "SRCI Contents entries count is not equal to SRCI Names entries count");

  for (size_t I : llvm::seq(SourceInfo->Contents.Parameters.Count))
    if (SourceInfo->Contents.Entries[I].Parameters.ContentSizeInBytes !=
        SourceInfo->Names.Entries[I].Parameters.ContentSizeInBytes)
      return parseFailed(formatv(
          "content size for entry {0} ({1} bytes) in SRCI Contents section "
          "does not match with size in SRCI Names section ({2} bytes)",
          I, SourceInfo->Contents.Entries[I].Parameters.ContentSizeInBytes,
          SourceInfo->Names.Entries[I].Parameters.ContentSizeInBytes));

  return Error::success();
}

Error DXContainer::parsePartOffsets() {
  uint32_t LastOffset =
      sizeof(dxbc::Header) + (Header.PartCount * sizeof(uint32_t));
  const char *Current = Data.getBuffer().data() + sizeof(dxbc::Header);
  for (uint32_t Part = 0; Part < Header.PartCount; ++Part) {
    uint32_t PartOffset;
    if (Error Err = readInteger(Data.getBuffer(), Current, PartOffset))
      return Err;
    if (PartOffset < LastOffset)
      return parseFailed(
          formatv(
              "Part offset for part {0} begins before the previous part ends",
              Part)
              .str());
    Current += sizeof(uint32_t);
    if (PartOffset >= Data.getBufferSize())
      return parseFailed("Part offset points beyond boundary of the file");
    // To prevent overflow when reading the part name, we subtract the part name
    // size from the buffer size, rather than adding to the offset. Since the
    // file header is larger than the part header we can't reach this code
    // unless the buffer is at least as large as a part header, so this
    // subtraction can't underflow.
    if (PartOffset >= Data.getBufferSize() - sizeof(dxbc::PartHeader::Name))
      return parseFailed("File not large enough to read part name");
    PartOffsets.push_back(PartOffset);

    dxbc::PartType PT =
        dxbc::parsePartType(Data.getBuffer().substr(PartOffset, 4));
    uint32_t PartDataStart = PartOffset + sizeof(dxbc::PartHeader);
    uint32_t PartSize;
    if (Error Err = readInteger(Data.getBuffer(),
                                Data.getBufferStart() + PartOffset + 4,
                                PartSize, "part size"))
      return Err;
    StringRef PartData = Data.getBuffer().substr(PartDataStart, PartSize);
    LastOffset = PartOffset + PartSize;
    switch (PT) {
    case dxbc::PartType::DXIL:
    case dxbc::PartType::ILDB:
      if (Error Err = parseDXILHeader(PT, PartData))
        return Err;
      break;
    case dxbc::PartType::ILDN:
      if (Error Err = parseDebugName(PartData))
        return Err;
      break;
    case dxbc::PartType::PRIV:
      if (Error Err = parsePrivateData(PartData))
        return Err;
      break;
    case dxbc::PartType::SFI0:
      if (Error Err = parseShaderFeatureFlags(PartData))
        return Err;
      break;
    case dxbc::PartType::HASH:
      if (Error Err = parseHash(PartData))
        return Err;
      break;
    case dxbc::PartType::PSV0:
      if (Error Err = parsePSVInfo(PartData))
        return Err;
      break;
    case dxbc::PartType::ISG1:
      if (Error Err = InputSignature.initialize(PartData))
        return Err;
      break;
    case dxbc::PartType::OSG1:
      if (Error Err = OutputSignature.initialize(PartData))
        return Err;
      break;
    case dxbc::PartType::PSG1:
      if (Error Err = PatchConstantSignature.initialize(PartData))
        return Err;
      break;
    case dxbc::PartType::Unknown:
      break;
    case dxbc::PartType::RTS0:
      if (Error Err = parseRootSignature(PartData))
        return Err;
      break;
    case dxbc::PartType::SRCI:
      if (Error Err = parseSourceInfo(PartData))
        return Err;
      break;
    case dxbc::PartType::VERS:
      if (Error Err = parseCompilerVersionInfo(PartData))
        return Err;
      break;
    }
  }

  if (DXIL && DebugDXIL &&
      DXIL->first.ShaderKind != DebugDXIL->first.ShaderKind)
    return parseFailed(
        "ILDB part shader kind does not match DXIL part shader kind");

  // Fully parsing the PSVInfo requires knowing the shader kind which we read
  // out of the program header in the DXIL part.
  if (PSVInfo) {
    std::optional<uint16_t> ShaderKind = getShaderKind();
    if (!ShaderKind)
      return parseFailed("cannot fully parse pipeline state validation "
                         "information without DXIL or ILDB part");
    if (Error Err = PSVInfo->parse(*ShaderKind))
      return Err;
  }
  return Error::success();
}

Expected<DXContainer> DXContainer::create(MemoryBufferRef Object) {
  DXContainer Container(Object);
  if (Error Err = Container.parseHeader())
    return std::move(Err);
  if (Error Err = Container.parsePartOffsets())
    return std::move(Err);
  return Container;
}

void DXContainer::PartIterator::updateIteratorImpl(const uint32_t Offset) {
  StringRef Buffer = Container.Data.getBuffer();
  const char *Current = Buffer.data() + Offset;
  // Offsets are validated during parsing, so all offsets in the container are
  // valid and contain enough readable data to read a header.
  cantFail(readStruct(Buffer, Current, IteratorState.Part));
  IteratorState.Data =
      StringRef(Current + sizeof(dxbc::PartHeader), IteratorState.Part.Size);
  IteratorState.Offset = Offset;
}

Error DirectX::RootSignature::parse() {
  const char *Current = PartData.begin();

  // Root Signature headers expects 6 integers to be present.
  if (PartData.size() < 6 * sizeof(uint32_t))
    return parseFailed(
        "Invalid root signature, insufficient space for header.");

  Version = support::endian::read<uint32_t, llvm::endianness::little>(Current);
  Current += sizeof(uint32_t);

  NumParameters =
      support::endian::read<uint32_t, llvm::endianness::little>(Current);
  Current += sizeof(uint32_t);

  RootParametersOffset =
      support::endian::read<uint32_t, llvm::endianness::little>(Current);
  Current += sizeof(uint32_t);

  NumStaticSamplers =
      support::endian::read<uint32_t, llvm::endianness::little>(Current);
  Current += sizeof(uint32_t);

  StaticSamplersOffset =
      support::endian::read<uint32_t, llvm::endianness::little>(Current);
  Current += sizeof(uint32_t);

  Flags = support::endian::read<uint32_t, llvm::endianness::little>(Current);
  Current += sizeof(uint32_t);

  ParametersHeaders.Data = PartData.substr(
      RootParametersOffset,
      NumParameters * sizeof(dxbc::RTS0::v1::RootParameterHeader));

  StaticSamplers.Stride = (Version <= 2)
                              ? sizeof(dxbc::RTS0::v1::StaticSampler)
                              : sizeof(dxbc::RTS0::v3::StaticSampler);

  StaticSamplers.Data = PartData.substr(StaticSamplersOffset,
                                        static_cast<size_t>(NumStaticSamplers) *
                                            StaticSamplers.Stride);

  return Error::success();
}

Error DirectX::PSVRuntimeInfo::parse(uint16_t ShaderKind) {
  Triple::EnvironmentType ShaderStage = dxbc::getShaderStage(ShaderKind);

  const char *Current = Data.begin();
  if (Error Err = readInteger(Data, Current, Size))
    return Err;
  Current += sizeof(uint32_t);

  StringRef PSVInfoData = Data.substr(sizeof(uint32_t), Size);

  if (PSVInfoData.size() < Size)
    return parseFailed(
        "Pipeline state data extends beyond the bounds of the part");

  using namespace dxbc::PSV;

  const uint32_t PSVVersion = getVersion();

  // Detect the PSVVersion by looking at the size field.
  if (PSVVersion == 3) {
    v3::RuntimeInfo Info;
    if (Error Err = readStruct(PSVInfoData, Current, Info))
      return Err;
    if (sys::IsBigEndianHost)
      Info.swapBytes(ShaderStage);
    BasicInfo = Info;
  } else if (PSVVersion == 2) {
    v2::RuntimeInfo Info;
    if (Error Err = readStruct(PSVInfoData, Current, Info))
      return Err;
    if (sys::IsBigEndianHost)
      Info.swapBytes(ShaderStage);
    BasicInfo = Info;
  } else if (PSVVersion == 1) {
    v1::RuntimeInfo Info;
    if (Error Err = readStruct(PSVInfoData, Current, Info))
      return Err;
    if (sys::IsBigEndianHost)
      Info.swapBytes(ShaderStage);
    BasicInfo = Info;
  } else if (PSVVersion == 0) {
    v0::RuntimeInfo Info;
    if (Error Err = readStruct(PSVInfoData, Current, Info))
      return Err;
    if (sys::IsBigEndianHost)
      Info.swapBytes(ShaderStage);
    BasicInfo = Info;
  } else
    return parseFailed(
        "Cannot read PSV Runtime Info, unsupported PSV version.");

  Current += Size;

  uint32_t ResourceCount = 0;
  if (Error Err = readInteger(Data, Current, ResourceCount))
    return Err;
  Current += sizeof(uint32_t);

  if (ResourceCount > 0) {
    if (Error Err = readInteger(Data, Current, Resources.Stride))
      return Err;
    Current += sizeof(uint32_t);

    size_t BindingDataSize = Resources.Stride * ResourceCount;
    Resources.Data = Data.substr(Current - Data.begin(), BindingDataSize);

    if (Resources.Data.size() < BindingDataSize)
      return parseFailed(
          "Resource binding data extends beyond the bounds of the part");

    Current += BindingDataSize;
  } else
    Resources.Stride = sizeof(v2::ResourceBindInfo);

  // PSV version 0 ends after the resource bindings.
  if (PSVVersion == 0)
    return Error::success();

  // String table starts at a 4-byte offset.
  Current = reinterpret_cast<const char *>(
      alignTo<dxbc::DXCONTAINER_STRUCT_ALIGNMENT>(
          reinterpret_cast<uintptr_t>(Current)));

  uint32_t StringTableSize = 0;
  if (Error Err = readInteger(Data, Current, StringTableSize))
    return Err;
  if (StringTableSize % 4 != 0)
    return parseFailed("String table misaligned");
  Current += sizeof(uint32_t);
  StringTable = StringRef(Current, StringTableSize);

  Current += StringTableSize;

  uint32_t SemanticIndexTableSize = 0;
  if (Error Err = readInteger(Data, Current, SemanticIndexTableSize))
    return Err;
  Current += sizeof(uint32_t);

  SemanticIndexTable.reserve(SemanticIndexTableSize);
  for (uint32_t I = 0; I < SemanticIndexTableSize; ++I) {
    uint32_t Index = 0;
    if (Error Err = readInteger(Data, Current, Index))
      return Err;
    Current += sizeof(uint32_t);
    SemanticIndexTable.push_back(Index);
  }

  uint8_t InputCount = getSigInputCount();
  uint8_t OutputCount = getSigOutputCount();
  uint8_t PatchOrPrimCount = getSigPatchOrPrimCount();

  uint32_t ElementCount = InputCount + OutputCount + PatchOrPrimCount;

  if (ElementCount > 0) {
    if (Error Err = readInteger(Data, Current, SigInputElements.Stride))
      return Err;
    Current += sizeof(uint32_t);
    // Assign the stride to all the arrays.
    SigOutputElements.Stride = SigPatchOrPrimElements.Stride =
        SigInputElements.Stride;

    if (Data.end() - Current <
        (ptrdiff_t)(ElementCount * SigInputElements.Stride))
      return parseFailed(
          "Signature elements extend beyond the size of the part");

    size_t InputSize = SigInputElements.Stride * InputCount;
    SigInputElements.Data = Data.substr(Current - Data.begin(), InputSize);
    Current += InputSize;

    size_t OutputSize = SigOutputElements.Stride * OutputCount;
    SigOutputElements.Data = Data.substr(Current - Data.begin(), OutputSize);
    Current += OutputSize;

    size_t PSize = SigPatchOrPrimElements.Stride * PatchOrPrimCount;
    SigPatchOrPrimElements.Data = Data.substr(Current - Data.begin(), PSize);
    Current += PSize;
  }

  ArrayRef<uint8_t> OutputVectorCounts = getOutputVectorCounts();
  uint8_t PatchConstOrPrimVectorCount = getPatchConstOrPrimVectorCount();
  uint8_t InputVectorCount = getInputVectorCount();

  auto maskDwordSize = [](uint8_t Vector) {
    return (static_cast<uint32_t>(Vector) + 7) >> 3;
  };

  auto mapTableSize = [maskDwordSize](uint8_t X, uint8_t Y) {
    return maskDwordSize(Y) * X * 4;
  };

  if (usesViewID()) {
    for (uint32_t I = 0; I < OutputVectorCounts.size(); ++I) {
      // The vector mask is one bit per component and 4 components per vector.
      // We can compute the number of dwords required by rounding up to the next
      // multiple of 8.
      uint32_t NumDwords =
          maskDwordSize(static_cast<uint32_t>(OutputVectorCounts[I]));
      size_t NumBytes = NumDwords * sizeof(uint32_t);
      OutputVectorMasks[I].Data = Data.substr(Current - Data.begin(), NumBytes);
      Current += NumBytes;
    }

    if (ShaderStage == Triple::Hull && PatchConstOrPrimVectorCount > 0) {
      uint32_t NumDwords = maskDwordSize(PatchConstOrPrimVectorCount);
      size_t NumBytes = NumDwords * sizeof(uint32_t);
      PatchOrPrimMasks.Data = Data.substr(Current - Data.begin(), NumBytes);
      Current += NumBytes;
    }
  }

  // Input/Output mapping table
  for (uint32_t I = 0; I < OutputVectorCounts.size(); ++I) {
    if (InputVectorCount == 0 || OutputVectorCounts[I] == 0)
      continue;
    uint32_t NumDwords = mapTableSize(InputVectorCount, OutputVectorCounts[I]);
    size_t NumBytes = NumDwords * sizeof(uint32_t);
    InputOutputMap[I].Data = Data.substr(Current - Data.begin(), NumBytes);
    Current += NumBytes;
  }

  // Hull shader: Input/Patch mapping table
  if (ShaderStage == Triple::Hull && PatchConstOrPrimVectorCount > 0 &&
      InputVectorCount > 0) {
    uint32_t NumDwords =
        mapTableSize(InputVectorCount, PatchConstOrPrimVectorCount);
    size_t NumBytes = NumDwords * sizeof(uint32_t);
    InputPatchMap.Data = Data.substr(Current - Data.begin(), NumBytes);
    Current += NumBytes;
  }

  // Domain Shader: Patch/Output mapping table
  if (ShaderStage == Triple::Domain && PatchConstOrPrimVectorCount > 0 &&
      OutputVectorCounts[0] > 0) {
    uint32_t NumDwords =
        mapTableSize(PatchConstOrPrimVectorCount, OutputVectorCounts[0]);
    size_t NumBytes = NumDwords * sizeof(uint32_t);
    PatchOutputMap.Data = Data.substr(Current - Data.begin(), NumBytes);
    Current += NumBytes;
  }

  return Error::success();
}

uint8_t DirectX::PSVRuntimeInfo::getSigInputCount() const {
  if (const auto *P = std::get_if<dxbc::PSV::v3::RuntimeInfo>(&BasicInfo))
    return P->SigInputElements;
  if (const auto *P = std::get_if<dxbc::PSV::v2::RuntimeInfo>(&BasicInfo))
    return P->SigInputElements;
  if (const auto *P = std::get_if<dxbc::PSV::v1::RuntimeInfo>(&BasicInfo))
    return P->SigInputElements;
  return 0;
}

uint8_t DirectX::PSVRuntimeInfo::getSigOutputCount() const {
  if (const auto *P = std::get_if<dxbc::PSV::v3::RuntimeInfo>(&BasicInfo))
    return P->SigOutputElements;
  if (const auto *P = std::get_if<dxbc::PSV::v2::RuntimeInfo>(&BasicInfo))
    return P->SigOutputElements;
  if (const auto *P = std::get_if<dxbc::PSV::v1::RuntimeInfo>(&BasicInfo))
    return P->SigOutputElements;
  return 0;
}

uint8_t DirectX::PSVRuntimeInfo::getSigPatchOrPrimCount() const {
  if (const auto *P = std::get_if<dxbc::PSV::v3::RuntimeInfo>(&BasicInfo))
    return P->SigPatchOrPrimElements;
  if (const auto *P = std::get_if<dxbc::PSV::v2::RuntimeInfo>(&BasicInfo))
    return P->SigPatchOrPrimElements;
  if (const auto *P = std::get_if<dxbc::PSV::v1::RuntimeInfo>(&BasicInfo))
    return P->SigPatchOrPrimElements;
  return 0;
}

class DXNotSupportedError : public ErrorInfo<DXNotSupportedError> {
public:
  static char ID;

  DXNotSupportedError(StringRef S) : FeatureString(S) {}

  void log(raw_ostream &OS) const override {
    OS << "DXContainer does not support " << FeatureString;
  }

  std::error_code convertToErrorCode() const override {
    return inconvertibleErrorCode();
  }

private:
  StringRef FeatureString;
};

char DXNotSupportedError::ID = 0;

Expected<section_iterator>
DXContainerObjectFile::getSymbolSection(DataRefImpl Symb) const {
  return make_error<DXNotSupportedError>("Symbol sections");
}

Expected<StringRef> DXContainerObjectFile::getSymbolName(DataRefImpl) const {
  return make_error<DXNotSupportedError>("Symbol names");
}

Expected<uint64_t>
DXContainerObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  return make_error<DXNotSupportedError>("Symbol addresses");
}

uint64_t DXContainerObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  llvm_unreachable("DXContainer does not support symbols");
}
uint64_t
DXContainerObjectFile::getCommonSymbolSizeImpl(DataRefImpl Symb) const {
  llvm_unreachable("DXContainer does not support symbols");
}

Expected<SymbolRef::Type>
DXContainerObjectFile::getSymbolType(DataRefImpl Symb) const {
  return make_error<DXNotSupportedError>("Symbol types");
}

void DXContainerObjectFile::moveSectionNext(DataRefImpl &Sec) const {
  PartIterator It = reinterpret_cast<PartIterator>(Sec.p);
  if (It == Parts.end())
    return;

  ++It;
  Sec.p = reinterpret_cast<uintptr_t>(It);
}

Expected<StringRef>
DXContainerObjectFile::getSectionName(DataRefImpl Sec) const {
  PartIterator It = reinterpret_cast<PartIterator>(Sec.p);
  return StringRef(It->Part.getName());
}

uint64_t DXContainerObjectFile::getSectionAddress(DataRefImpl Sec) const {
  PartIterator It = reinterpret_cast<PartIterator>(Sec.p);
  return It->Offset;
}

uint64_t DXContainerObjectFile::getSectionIndex(DataRefImpl Sec) const {
  return (Sec.p - reinterpret_cast<uintptr_t>(Parts.begin())) /
         sizeof(PartIterator);
}

uint64_t DXContainerObjectFile::getSectionSize(DataRefImpl Sec) const {
  PartIterator It = reinterpret_cast<PartIterator>(Sec.p);
  return It->Data.size();
}
Expected<ArrayRef<uint8_t>>
DXContainerObjectFile::getSectionContents(DataRefImpl Sec) const {
  PartIterator It = reinterpret_cast<PartIterator>(Sec.p);
  return ArrayRef<uint8_t>(It->Data.bytes_begin(), It->Data.size());
}

uint64_t DXContainerObjectFile::getSectionAlignment(DataRefImpl Sec) const {
  return 1;
}

bool DXContainerObjectFile::isSectionCompressed(DataRefImpl Sec) const {
  return false;
}

bool DXContainerObjectFile::isSectionText(DataRefImpl Sec) const {
  return false;
}

bool DXContainerObjectFile::isSectionData(DataRefImpl Sec) const {
  return false;
}

bool DXContainerObjectFile::isSectionBSS(DataRefImpl Sec) const {
  return false;
}

bool DXContainerObjectFile::isSectionVirtual(DataRefImpl Sec) const {
  return false;
}

relocation_iterator
DXContainerObjectFile::section_rel_begin(DataRefImpl Sec) const {
  return relocation_iterator(RelocationRef());
}

relocation_iterator
DXContainerObjectFile::section_rel_end(DataRefImpl Sec) const {
  return relocation_iterator(RelocationRef());
}

void DXContainerObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  llvm_unreachable("DXContainer does not support relocations");
}

uint64_t DXContainerObjectFile::getRelocationOffset(DataRefImpl Rel) const {
  llvm_unreachable("DXContainer does not support relocations");
}

symbol_iterator
DXContainerObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  return symbol_iterator(SymbolRef());
}

uint64_t DXContainerObjectFile::getRelocationType(DataRefImpl Rel) const {
  llvm_unreachable("DXContainer does not support relocations");
}

void DXContainerObjectFile::getRelocationTypeName(
    DataRefImpl Rel, SmallVectorImpl<char> &Result) const {
  llvm_unreachable("DXContainer does not support relocations");
}

section_iterator DXContainerObjectFile::section_begin() const {
  DataRefImpl Sec;
  Sec.p = reinterpret_cast<uintptr_t>(Parts.begin());
  return section_iterator(SectionRef(Sec, this));
}
section_iterator DXContainerObjectFile::section_end() const {
  DataRefImpl Sec;
  Sec.p = reinterpret_cast<uintptr_t>(Parts.end());
  return section_iterator(SectionRef(Sec, this));
}

uint8_t DXContainerObjectFile::getBytesInAddress() const { return 4; }

StringRef DXContainerObjectFile::getFileFormatName() const {
  return "DirectX Container";
}

Triple::ArchType DXContainerObjectFile::getArch() const { return Triple::dxil; }

Expected<SubtargetFeatures> DXContainerObjectFile::getFeatures() const {
  return SubtargetFeatures();
}

Error DXContainerObjectFile::printSymbolName(raw_ostream &OS,
                                             DataRefImpl Symb) const {
  return make_error<DXNotSupportedError>("Symbol names");
}

Expected<uint32_t>
DXContainerObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  return make_error<DXNotSupportedError>("Symbol flags");
}

Expected<std::unique_ptr<DXContainerObjectFile>>
ObjectFile::createDXContainerObjectFile(MemoryBufferRef Object) {
  auto ExC = DXContainer::create(Object);
  if (!ExC)
    return ExC.takeError();
  std::unique_ptr<DXContainerObjectFile> Obj(new DXContainerObjectFile(*ExC));
  return std::move(Obj);
}
