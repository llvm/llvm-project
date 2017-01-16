//===- Trace.cpp - XRay Trace Loading implementation. ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// XRay log reader implementation.
//
//===----------------------------------------------------------------------===//
#include "llvm/XRay/Trace.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/XRay/YAMLXRayRecord.h"

using namespace llvm;
using namespace llvm::xray;
using llvm::yaml::Input;

using XRayRecordStorage =
    std::aligned_storage<sizeof(XRayRecord), alignof(XRayRecord)>::type;

Error NaiveLogLoader(StringRef Data, XRayFileHeader &FileHeader,
                     std::vector<XRayRecord> &Records) {
  // FIXME: Maybe deduce whether the data is little or big-endian using some
  // magic bytes in the beginning of the file?

  // First 32 bytes of the file will always be the header. We assume a certain
  // format here:
  //
  //   (2)   uint16 : version
  //   (2)   uint16 : type
  //   (4)   uint32 : bitfield
  //   (8)   uint64 : cycle frequency
  //   (16)  -      : padding
  //
  if (Data.size() < 32)
    return make_error<StringError>(
        "Not enough bytes for an XRay log.",
        std::make_error_code(std::errc::invalid_argument));

  if (Data.size() - 32 == 0 || Data.size() % 32 != 0)
    return make_error<StringError>(
        "Invalid-sized XRay data.",
        std::make_error_code(std::errc::invalid_argument));

  DataExtractor HeaderExtractor(Data, true, 8);
  uint32_t OffsetPtr = 0;
  FileHeader.Version = HeaderExtractor.getU16(&OffsetPtr);
  FileHeader.Type = HeaderExtractor.getU16(&OffsetPtr);
  uint32_t Bitfield = HeaderExtractor.getU32(&OffsetPtr);
  FileHeader.ConstantTSC = Bitfield & 1uL;
  FileHeader.NonstopTSC = Bitfield & 1uL << 1;
  FileHeader.CycleFrequency = HeaderExtractor.getU64(&OffsetPtr);

  if (FileHeader.Version != 1)
    return make_error<StringError>(
        Twine("Unsupported XRay file version: ") + Twine(FileHeader.Version),
        std::make_error_code(std::errc::invalid_argument));

  // Each record after the header will be 32 bytes, in the following format:
  //
  //   (2)   uint16 : record type
  //   (1)   uint8  : cpu id
  //   (1)   uint8  : type
  //   (4)   sint32 : function id
  //   (8)   uint64 : tsc
  //   (4)   uint32 : thread id
  //   (12)  -      : padding
  for (auto S = Data.drop_front(32); !S.empty(); S = S.drop_front(32)) {
    DataExtractor RecordExtractor(S, true, 8);
    uint32_t OffsetPtr = 0;
    Records.emplace_back();
    auto &Record = Records.back();
    Record.RecordType = RecordExtractor.getU16(&OffsetPtr);
    Record.CPU = RecordExtractor.getU8(&OffsetPtr);
    auto Type = RecordExtractor.getU8(&OffsetPtr);
    switch (Type) {
    case 0:
      Record.Type = RecordTypes::ENTER;
      break;
    case 1:
      Record.Type = RecordTypes::EXIT;
      break;
    default:
      return make_error<StringError>(
          Twine("Unknown record type '") + Twine(int{Type}) + "'",
          std::make_error_code(std::errc::executable_format_error));
    }
    Record.FuncId = RecordExtractor.getSigned(&OffsetPtr, sizeof(int32_t));
    Record.TSC = RecordExtractor.getU64(&OffsetPtr);
    Record.TId = RecordExtractor.getU32(&OffsetPtr);
  }
  return Error::success();
}

Error YAMLLogLoader(StringRef Data, XRayFileHeader &FileHeader,
                    std::vector<XRayRecord> &Records) {

  // Load the documents from the MappedFile.
  YAMLXRayTrace Trace;
  Input In(Data);
  In >> Trace;
  if (In.error())
    return make_error<StringError>("Failed loading YAML Data.", In.error());

  FileHeader.Version = Trace.Header.Version;
  FileHeader.Type = Trace.Header.Type;
  FileHeader.ConstantTSC = Trace.Header.ConstantTSC;
  FileHeader.NonstopTSC = Trace.Header.NonstopTSC;
  FileHeader.CycleFrequency = Trace.Header.CycleFrequency;

  if (FileHeader.Version != 1)
    return make_error<StringError>(
        Twine("Unsupported XRay file version: ") + Twine(FileHeader.Version),
        std::make_error_code(std::errc::invalid_argument));

  Records.clear();
  std::transform(Trace.Records.begin(), Trace.Records.end(),
                 std::back_inserter(Records), [&](const YAMLXRayRecord &R) {
                   return XRayRecord{R.RecordType, R.CPU, R.Type,
                                     R.FuncId,     R.TSC, R.TId};
                 });
  return Error::success();
}

Expected<Trace> llvm::xray::loadTraceFile(StringRef Filename, bool Sort) {
  int Fd;
  if (auto EC = sys::fs::openFileForRead(Filename, Fd)) {
    return make_error<StringError>(
        Twine("Cannot read log from '") + Filename + "'", EC);
  }

  // Attempt to get the filesize.
  uint64_t FileSize;
  if (auto EC = sys::fs::file_size(Filename, FileSize)) {
    return make_error<StringError>(
        Twine("Cannot read log from '") + Filename + "'", EC);
  }
  if (FileSize < 4) {
    return make_error<StringError>(
        Twine("File '") + Filename + "' too small for XRay.",
        std::make_error_code(std::errc::executable_format_error));
  }

  // Attempt to mmap the file.
  std::error_code EC;
  sys::fs::mapped_file_region MappedFile(
      Fd, sys::fs::mapped_file_region::mapmode::readonly, FileSize, 0, EC);
  if (EC) {
    return make_error<StringError>(
        Twine("Cannot read log from '") + Filename + "'", EC);
  }

  // Attempt to detect the file type using file magic. We have a slight bias
  // towards the binary format, and we do this by making sure that the first 4
  // bytes of the binary file is some combination of the following byte
  // patterns:
  //
  //   0x0001 0x0000 - version 1, "naive" format
  //   0x0001 0x0001 - version 1, "flight data recorder" format
  //
  // YAML files dont' typically have those first four bytes as valid text so we
  // try loading assuming YAML if we don't find these bytes.
  //
  // Only if we can't load either the binary or the YAML format will we yield an
  // error.
  StringRef Magic(MappedFile.data(), 4);
  DataExtractor HeaderExtractor(Magic, true, 8);
  uint32_t OffsetPtr = 0;
  uint16_t Version = HeaderExtractor.getU16(&OffsetPtr);
  uint16_t Type = HeaderExtractor.getU16(&OffsetPtr);

  Trace T;
  if (Version == 1 && (Type == 0 || Type == 1)) {
    if (auto E = NaiveLogLoader(StringRef(MappedFile.data(), MappedFile.size()),
                                T.FileHeader, T.Records))
      return std::move(E);
  } else {
    if (auto E = YAMLLogLoader(StringRef(MappedFile.data(), MappedFile.size()),
                               T.FileHeader, T.Records))
      return std::move(E);
  }

  if (Sort)
    std::sort(T.Records.begin(), T.Records.end(),
              [&](const XRayRecord &L, const XRayRecord &R) {
                return L.TSC < R.TSC;
              });

  return std::move(T);
}
