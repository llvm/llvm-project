//===- BitstreamRemarkParser.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility methods used by clients that want to use the
// parser for remark diagnostics in LLVM.
//
//===----------------------------------------------------------------------===//

#include "BitstreamRemarkParser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <optional>

using namespace llvm;
using namespace llvm::remarks;

namespace {

template <typename... Ts> Error error(char const *Fmt, const Ts &...Vals) {
  std::string Buffer;
  raw_string_ostream OS(Buffer);
  OS << formatv(Fmt, Vals...);
  return make_error<StringError>(
      std::move(Buffer),
      std::make_error_code(std::errc::illegal_byte_sequence));
}

} // namespace

Error BitstreamBlockParserHelperBase::unknownRecord(unsigned AbbrevID) {
  return error("Unknown record entry ({}).", AbbrevID);
}

Error BitstreamBlockParserHelperBase::unexpectedRecord(StringRef RecordName) {
  return error("Unexpected record entry ({}).", RecordName);
}

Error BitstreamBlockParserHelperBase::malformedRecord(StringRef RecordName) {
  return error("Malformed record entry ({}).", RecordName);
}

Error BitstreamBlockParserHelperBase::unexpectedBlock(unsigned Code) {
  return error("Unexpected subblock ({}).", Code);
}

static Expected<unsigned> expectSubBlock(BitstreamCursor &Stream) {
  Expected<BitstreamEntry> Next = Stream.advance();
  if (!Next)
    return Next.takeError();
  switch (Next->Kind) {
  case BitstreamEntry::SubBlock:
    return Next->ID;
  case BitstreamEntry::Record:
  case BitstreamEntry::EndBlock:
    return error("Expected subblock, but got unexpected record.");
  case BitstreamEntry::Error:
    return error("Expected subblock, but got unexpected end of bitstream.");
  }
  llvm_unreachable("Unexpected BitstreamEntry");
}

Error BitstreamBlockParserHelperBase::expectBlock() {
  auto MaybeBlockID = expectSubBlock(Stream);
  if (!MaybeBlockID)
    return MaybeBlockID.takeError();
  if (*MaybeBlockID != BlockID)
    return error("Expected {} block, but got unexpected block ({}).", BlockName,
                 *MaybeBlockID);
  return Error::success();
}

Error BitstreamBlockParserHelperBase::enterBlock() {
  if (Stream.EnterSubBlock(BlockID))
    return error("Error while entering {} block.", BlockName);
  return Error::success();
}

Error BitstreamMetaParserHelper::parseRecord(unsigned Code) {
  // Note: 2 is used here because it's the max number of fields we have per
  // record.
  SmallVector<uint64_t, 2> Record;
  StringRef Blob;
  Expected<unsigned> RecordID = Stream.readRecord(Code, Record, &Blob);
  if (!RecordID)
    return RecordID.takeError();

  switch (*RecordID) {
  case RECORD_META_CONTAINER_INFO: {
    if (Record.size() != 2)
      return malformedRecord(MetaContainerInfoName);
    Container = {Record[0], Record[1]};
    // Error immediately if container version is outdated, so the user sees an
    // explanation instead of a parser error.
    if (Container->Version != CurrentContainerVersion) {
      return ::error(
          "Unsupported remark container version (expected: {}, read: {}). "
          "Please upgrade/downgrade your toolchain to read this container.",
          CurrentContainerVersion, Container->Version);
    }
    break;
  }
  case RECORD_META_REMARK_VERSION: {
    if (Record.size() != 1)
      return malformedRecord(MetaRemarkVersionName);
    RemarkVersion = Record[0];
    // Error immediately if remark version is outdated, so the user sees an
    // explanation instead of a parser error.
    if (*RemarkVersion != CurrentRemarkVersion) {
      return ::error(
          "Unsupported remark version in container (expected: {}, read: {}). "
          "Please upgrade/downgrade your toolchain to read this container.",
          CurrentRemarkVersion, *RemarkVersion);
    }
    break;
  }
  case RECORD_META_STRTAB: {
    if (Record.size() != 0)
      return malformedRecord(MetaStrTabName);
    StrTabBuf = Blob;
    break;
  }
  case RECORD_META_EXTERNAL_FILE: {
    if (Record.size() != 0)
      return malformedRecord(MetaExternalFileName);
    ExternalFilePath = Blob;
    break;
  }
  default:
    return unknownRecord(*RecordID);
  }
  return Error::success();
}

Error BitstreamRemarkParserHelper::parseRecord(unsigned Code) {
  Record.clear();
  Expected<unsigned> MaybeRecordID =
      Stream.readRecord(Code, Record, &RecordBlob);
  if (!MaybeRecordID)
    return MaybeRecordID.takeError();
  RecordID = *MaybeRecordID;
  return handleRecord();
}

Error BitstreamRemarkParserHelper::handleRecord() {
  switch (RecordID) {
  case RECORD_REMARK_HEADER: {
    if (Record.size() != 4)
      return malformedRecord(RemarkHeaderName);
    Type = Record[0];
    RemarkNameIdx = Record[1];
    PassNameIdx = Record[2];
    FunctionNameIdx = Record[3];
    break;
  }
  case RECORD_REMARK_DEBUG_LOC: {
    if (Record.size() != 3)
      return malformedRecord(RemarkDebugLocName);
    Loc = {Record[0], Record[1], Record[2]};
    break;
  }
  case RECORD_REMARK_HOTNESS: {
    if (Record.size() != 1)
      return malformedRecord(RemarkHotnessName);
    Hotness = Record[0];
    break;
  }
  case RECORD_REMARK_ARG_WITH_DEBUGLOC: {
    if (Record.size() != 5)
      return malformedRecord(RemarkArgWithDebugLocName);
    auto &Arg = Args.emplace_back(Record[0], Record[1]);
    Arg.Loc = {Record[2], Record[3], Record[4]};
    break;
  }
  case RECORD_REMARK_ARG_WITHOUT_DEBUGLOC: {
    if (Record.size() != 2)
      return malformedRecord(RemarkArgWithoutDebugLocName);
    Args.emplace_back(Record[0], Record[1]);
    break;
  }
  default:
    return unknownRecord(RecordID);
  }
  return Error::success();
}

Error BitstreamRemarkParserHelper::parseNext() {
  Type.reset();
  RemarkNameIdx.reset();
  PassNameIdx.reset();
  FunctionNameIdx.reset();
  Hotness.reset();
  Loc.reset();
  Args.clear();

  if (Error E = expectBlock())
    return E;
  return parseBlock();
}

BitstreamParserHelper::BitstreamParserHelper(StringRef Buffer)
    : Stream(Buffer) {}

Error BitstreamParserHelper::expectMagic() {
  std::array<char, 4> Result;
  for (unsigned I = 0; I < 4; ++I)
    if (Expected<unsigned> R = Stream.Read(8))
      Result[I] = *R;
    else
      return R.takeError();

  StringRef MagicNumber{Result.data(), Result.size()};
  if (MagicNumber != remarks::ContainerMagic)
    return error("Unknown magic number: expecting {}, got {}.",
                 remarks::ContainerMagic, MagicNumber);
  return Error::success();
}

Error BitstreamParserHelper::parseBlockInfoBlock() {
  Expected<BitstreamEntry> Next = Stream.advance();
  if (!Next)
    return Next.takeError();
  if (Next->Kind != BitstreamEntry::SubBlock ||
      Next->ID != llvm::bitc::BLOCKINFO_BLOCK_ID)
    return error(
        "Error while parsing BLOCKINFO_BLOCK: expecting [ENTER_SUBBLOCK, "
        "BLOCKINFO_BLOCK, ...].");

  Expected<std::optional<BitstreamBlockInfo>> MaybeBlockInfo =
      Stream.ReadBlockInfoBlock();
  if (!MaybeBlockInfo)
    return MaybeBlockInfo.takeError();

  if (!*MaybeBlockInfo)
    return error("Missing BLOCKINFO_BLOCK.");

  BlockInfo = **MaybeBlockInfo;

  Stream.setBlockInfo(&BlockInfo);
  return Error::success();
}

Error BitstreamParserHelper::advanceToMetaBlock() {
  if (Error E = expectMagic())
    return E;
  if (Error E = parseBlockInfoBlock())
    return E;
  return Error::success();
}

Expected<std::unique_ptr<BitstreamRemarkParser>>
remarks::createBitstreamParserFromMeta(
    StringRef Buf, std::optional<StringRef> ExternalFilePrependPath) {
  auto Parser = std::make_unique<BitstreamRemarkParser>(Buf);

  if (ExternalFilePrependPath)
    Parser->ExternalFilePrependPath = std::string(*ExternalFilePrependPath);

  return std::move(Parser);
}

Expected<std::unique_ptr<Remark>> BitstreamRemarkParser::next() {
  if (ParserHelper.atEndOfStream())
    return make_error<EndOfFileError>();

  if (!ReadyToParseRemarks) {
    if (Error E = parseMeta())
      return std::move(E);
    ReadyToParseRemarks = true;
  }

  return parseRemark();
}

Error BitstreamRemarkParser::parseMeta() {
  if (Error E = ParserHelper.advanceToMetaBlock())
    return E;

  BitstreamMetaParserHelper MetaHelper(ParserHelper.Stream);
  if (Error E = MetaHelper.expectBlock())
    return E;
  if (Error E = MetaHelper.parseBlock())
    return E;

  if (Error E = processCommonMeta(MetaHelper))
    return E;

  switch (ContainerType) {
  case BitstreamRemarkContainerType::Standalone:
    return processStandaloneMeta(MetaHelper);
  case BitstreamRemarkContainerType::SeparateRemarksFile:
    return processSeparateRemarksFileMeta(MetaHelper);
  case BitstreamRemarkContainerType::SeparateRemarksMeta:
    return processSeparateRemarksMetaMeta(MetaHelper);
  }
  llvm_unreachable("Unknown BitstreamRemarkContainerType enum");
}

Error BitstreamRemarkParser::processCommonMeta(
    BitstreamMetaParserHelper &Helper) {
  if (!Helper.Container)
    return Helper.error("Missing container info.");
  auto &Container = *Helper.Container;
  ContainerVersion = Container.Version;
  // Always >= BitstreamRemarkContainerType::First since it's unsigned.
  if (Container.Type > static_cast<uint8_t>(BitstreamRemarkContainerType::Last))
    return Helper.error("Invalid container type.");
  ContainerType = static_cast<BitstreamRemarkContainerType>(Container.Type);
  return Error::success();
}

Error BitstreamRemarkParser::processStrTab(BitstreamMetaParserHelper &Helper) {
  if (!Helper.StrTabBuf)
    return Helper.error("Missing string table.");
  // Parse and assign the string table.
  StrTab.emplace(*Helper.StrTabBuf);
  return Error::success();
}

Error BitstreamRemarkParser::processRemarkVersion(
    BitstreamMetaParserHelper &Helper) {
  if (!Helper.RemarkVersion)
    return Helper.error("Missing remark version.");
  RemarkVersion = *Helper.RemarkVersion;
  return Error::success();
}

Error BitstreamRemarkParser::processExternalFilePath(
    BitstreamMetaParserHelper &Helper) {
  if (!Helper.ExternalFilePath)
    return Helper.error("Missing external file path.");
  StringRef ExternalFilePath = *Helper.ExternalFilePath;

  SmallString<80> FullPath(ExternalFilePrependPath);
  sys::path::append(FullPath, ExternalFilePath);

  // External file: open the external file, parse it, check if its metadata
  // matches the one from the separate metadata, then replace the current parser
  // with the one parsing the remarks.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(FullPath);
  if (std::error_code EC = BufferOrErr.getError())
    return createFileError(FullPath, EC);

  TmpRemarkBuffer = std::move(*BufferOrErr);

  // Don't try to parse the file if it's empty.
  if (TmpRemarkBuffer->getBufferSize() == 0)
    return make_error<EndOfFileError>();

  // Create a separate parser used for parsing the separate file.
  ParserHelper = BitstreamParserHelper(TmpRemarkBuffer->getBuffer());
  // Advance and check until we can parse the meta block.
  if (Error E = ParserHelper.advanceToMetaBlock())
    return E;
  // Parse the meta from the separate file.
  // Note: here we overwrite the BlockInfo with the one from the file. This will
  // be used to parse the rest of the file.
  BitstreamMetaParserHelper SeparateMetaHelper(ParserHelper.Stream);
  if (Error E = SeparateMetaHelper.expectBlock())
    return E;
  if (Error E = SeparateMetaHelper.parseBlock())
    return E;

  if (Error E = processCommonMeta(SeparateMetaHelper))
    return E;

  if (ContainerType != BitstreamRemarkContainerType::SeparateRemarksFile)
    return SeparateMetaHelper.error("Wrong container type in external file.");

  // Process the meta from the separate file.
  return processSeparateRemarksFileMeta(SeparateMetaHelper);
}

Error BitstreamRemarkParser::processStandaloneMeta(
    BitstreamMetaParserHelper &Helper) {
  if (Error E = processStrTab(Helper))
    return E;
  return processRemarkVersion(Helper);
}

Error BitstreamRemarkParser::processSeparateRemarksFileMeta(
    BitstreamMetaParserHelper &Helper) {
  return processRemarkVersion(Helper);
}

Error BitstreamRemarkParser::processSeparateRemarksMetaMeta(
    BitstreamMetaParserHelper &Helper) {
  if (Error E = processStrTab(Helper))
    return E;
  return processExternalFilePath(Helper);
}

Expected<std::unique_ptr<Remark>> BitstreamRemarkParser::parseRemark() {
  BitstreamRemarkParserHelper RemarkHelper(ParserHelper.Stream);
  if (Error E = RemarkHelper.parseNext())
    return std::move(E);

  return processRemark(RemarkHelper);
}

Expected<std::unique_ptr<Remark>>
BitstreamRemarkParser::processRemark(BitstreamRemarkParserHelper &Helper) {
  std::unique_ptr<Remark> Result = std::make_unique<Remark>();
  Remark &R = *Result;

  if (!StrTab)
    return Helper.error("Missing string table.");

  if (!Helper.Type)
    return Helper.error("Missing remark type.");

  // Always >= Type::First since it's unsigned.
  if (*Helper.Type > static_cast<uint8_t>(Type::Last))
    return Helper.error("Unknown remark type.");

  R.RemarkType = static_cast<Type>(*Helper.Type);

  if (!Helper.RemarkNameIdx)
    return Helper.error("Missing remark name.");

  if (Expected<StringRef> RemarkName = (*StrTab)[*Helper.RemarkNameIdx])
    R.RemarkName = *RemarkName;
  else
    return RemarkName.takeError();

  if (!Helper.PassNameIdx)
    return Helper.error("Missing remark pass.");

  if (Expected<StringRef> PassName = (*StrTab)[*Helper.PassNameIdx])
    R.PassName = *PassName;
  else
    return PassName.takeError();

  if (!Helper.FunctionNameIdx)
    return Helper.error("Missing remark function name.");

  if (Expected<StringRef> FunctionName = (*StrTab)[*Helper.FunctionNameIdx])
    R.FunctionName = *FunctionName;
  else
    return FunctionName.takeError();

  if (Helper.Loc) {
    Expected<StringRef> SourceFileName =
        (*StrTab)[Helper.Loc->SourceFileNameIdx];
    if (!SourceFileName)
      return SourceFileName.takeError();
    R.Loc.emplace();
    R.Loc->SourceFilePath = *SourceFileName;
    R.Loc->SourceLine = Helper.Loc->SourceLine;
    R.Loc->SourceColumn = Helper.Loc->SourceColumn;
  }

  if (Helper.Hotness)
    R.Hotness = *Helper.Hotness;

  for (const BitstreamRemarkParserHelper::Argument &Arg : Helper.Args) {
    if (!Arg.KeyIdx)
      return Helper.error("Missing key in remark argument.");
    if (!Arg.ValueIdx)
      return Helper.error("Missing value in remark argument.");

    // We have at least a key and a value, create an entry.
    auto &RArg = R.Args.emplace_back();

    if (Expected<StringRef> Key = (*StrTab)[*Arg.KeyIdx])
      RArg.Key = *Key;
    else
      return Key.takeError();

    if (Expected<StringRef> Value = (*StrTab)[*Arg.ValueIdx])
      RArg.Val = *Value;
    else
      return Value.takeError();

    if (Arg.Loc) {
      if (Expected<StringRef> SourceFileName =
              (*StrTab)[Arg.Loc->SourceFileNameIdx]) {
        RArg.Loc.emplace();
        RArg.Loc->SourceFilePath = *SourceFileName;
        RArg.Loc->SourceLine = Arg.Loc->SourceLine;
        RArg.Loc->SourceColumn = Arg.Loc->SourceColumn;
      } else
        return SourceFileName.takeError();
    }
  }

  return std::move(Result);
}
llvm::remarks::BitstreamRemarkParser::BitstreamRemarkParser(StringRef Buf)
    : RemarkParser(Format::Bitstream), ParserHelper(Buf) {}
