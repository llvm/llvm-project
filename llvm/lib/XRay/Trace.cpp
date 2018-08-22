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
#include "llvm/XRay/FileHeaderReader.h"
#include "llvm/XRay/YAMLXRayRecord.h"

using namespace llvm;
using namespace llvm::xray;
using llvm::yaml::Input;

namespace {
using XRayRecordStorage =
    std::aligned_storage<sizeof(XRayRecord), alignof(XRayRecord)>::type;

// This is the number of bytes in the "body" of a MetadataRecord in FDR Mode.
// This already excludes the first byte, which indicates the type of metadata
// record it is.
constexpr auto kFDRMetadataBodySize = 15;

Error loadNaiveFormatLog(StringRef Data, XRayFileHeader &FileHeader,
                         std::vector<XRayRecord> &Records) {
  if (Data.size() < 32)
    return make_error<StringError>(
        "Not enough bytes for an XRay log.",
        std::make_error_code(std::errc::invalid_argument));

  if (Data.size() - 32 == 0 || Data.size() % 32 != 0)
    return make_error<StringError>(
        "Invalid-sized XRay data.",
        std::make_error_code(std::errc::invalid_argument));

  DataExtractor Reader(Data, true, 8);
  uint32_t OffsetPtr = 0;
  auto FileHeaderOrError = readBinaryFormatHeader(Reader, OffsetPtr);
  if (!FileHeaderOrError)
    return FileHeaderOrError.takeError();
  FileHeader = std::move(FileHeaderOrError.get());

  // Each record after the header will be 32 bytes, in the following format:
  //
  //   (2)   uint16 : record type
  //   (1)   uint8  : cpu id
  //   (1)   uint8  : type
  //   (4)   sint32 : function id
  //   (8)   uint64 : tsc
  //   (4)   uint32 : thread id
  //   (4)   uint32 : process id
  //   (8)   -      : padding
  while (Reader.isValidOffset(OffsetPtr)) {
    if (!Reader.isValidOffsetForDataOfSize(OffsetPtr, 32))
      return createStringError(
          std::make_error_code(std::errc::executable_format_error),
          "Not enough bytes to read a full record at offset %d.", OffsetPtr);
    auto PreReadOffset = OffsetPtr;
    auto RecordType = Reader.getU16(&OffsetPtr);
    if (OffsetPtr == PreReadOffset)
      return createStringError(
          std::make_error_code(std::errc::executable_format_error),
          "Failed reading record type at offset %d.", OffsetPtr);

    switch (RecordType) {
    case 0: { // Normal records.
      Records.emplace_back();
      auto &Record = Records.back();
      Record.RecordType = RecordType;

      PreReadOffset = OffsetPtr;
      Record.CPU = Reader.getU8(&OffsetPtr);
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading CPU field at offset %d.", OffsetPtr);

      PreReadOffset = OffsetPtr;
      auto Type = Reader.getU8(&OffsetPtr);
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading record type field at offset %d.", OffsetPtr);

      switch (Type) {
      case 0:
        Record.Type = RecordTypes::ENTER;
        break;
      case 1:
        Record.Type = RecordTypes::EXIT;
        break;
      case 2:
        Record.Type = RecordTypes::TAIL_EXIT;
        break;
      case 3:
        Record.Type = RecordTypes::ENTER_ARG;
        break;
      default:
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Unknown record type '%d' at offset %d.", Type, OffsetPtr);
      }

      PreReadOffset = OffsetPtr;
      Record.FuncId = Reader.getSigned(&OffsetPtr, sizeof(int32_t));
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading function id field at offset %d.", OffsetPtr);

      PreReadOffset = OffsetPtr;
      Record.TSC = Reader.getU64(&OffsetPtr);
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading TSC field at offset %d.", OffsetPtr);

      PreReadOffset = OffsetPtr;
      Record.TId = Reader.getU32(&OffsetPtr);
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading thread id field at offset %d.", OffsetPtr);

      PreReadOffset = OffsetPtr;
      Record.PId = Reader.getU32(&OffsetPtr);
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading process id at offset %d.", OffsetPtr);

      break;
    }
    case 1: { // Arg payload record.
      auto &Record = Records.back();

      // We skip the next two bytes of the record, because we don't need the
      // type and the CPU record for arg payloads.
      OffsetPtr += 2;
      PreReadOffset = OffsetPtr;
      int32_t FuncId = Reader.getSigned(&OffsetPtr, sizeof(int32_t));
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading function id field at offset %d.", OffsetPtr);

      PreReadOffset = OffsetPtr;
      auto TId = Reader.getU32(&OffsetPtr);
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading thread id field at offset %d.", OffsetPtr);

      PreReadOffset = OffsetPtr;
      auto PId = Reader.getU32(&OffsetPtr);
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading process id field at offset %d.", OffsetPtr);

      // Make a check for versions above 3 for the Pid field
      if (Record.FuncId != FuncId || Record.TId != TId ||
          (FileHeader.Version >= 3 ? Record.PId != PId : false))
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Corrupted log, found arg payload following non-matching "
            "function+thread record. Record for function %d != %d at offset "
            "%d",
            Record.FuncId, FuncId, OffsetPtr);

      PreReadOffset = OffsetPtr;
      auto Arg = Reader.getU64(&OffsetPtr);
      if (OffsetPtr == PreReadOffset)
        return createStringError(
            std::make_error_code(std::errc::executable_format_error),
            "Failed reading argument payload at offset %d.", OffsetPtr);

      Record.CallArgs.push_back(Arg);
      break;
    }
    default:
      return createStringError(
          std::make_error_code(std::errc::executable_format_error),
          "Unknown record type '%d' at offset %d.", RecordType, OffsetPtr);
    }
    // Advance the offset pointer enough bytes to align to 32-byte records for
    // basic mode logs.
    OffsetPtr += 8;
  }
  return Error::success();
}

/// When reading from a Flight Data Recorder mode log, metadata records are
/// sparse compared to packed function records, so we must maintain state as we
/// read through the sequence of entries. This allows the reader to denormalize
/// the CPUId and Thread Id onto each Function Record and transform delta
/// encoded TSC values into absolute encodings on each record.
struct FDRState {
  uint16_t CPUId;
  uint16_t ThreadId;
  int32_t ProcessId;
  uint64_t BaseTSC;

  /// Encode some of the state transitions for the FDR log reader as explicit
  /// checks. These are expectations for the next Record in the stream.
  enum class Token {
    NEW_BUFFER_RECORD_OR_EOF,
    WALLCLOCK_RECORD,
    NEW_CPU_ID_RECORD,
    FUNCTION_SEQUENCE,
    SCAN_TO_END_OF_THREAD_BUF,
    CUSTOM_EVENT_DATA,
    CALL_ARGUMENT,
    BUFFER_EXTENTS,
    PID_RECORD,
  };
  Token Expects;

  // Each threads buffer may have trailing garbage to scan over, so we track our
  // progress.
  uint64_t CurrentBufferSize;
  uint64_t CurrentBufferConsumed;
};

const char *fdrStateToTwine(const FDRState::Token &state) {
  switch (state) {
  case FDRState::Token::NEW_BUFFER_RECORD_OR_EOF:
    return "NEW_BUFFER_RECORD_OR_EOF";
  case FDRState::Token::WALLCLOCK_RECORD:
    return "WALLCLOCK_RECORD";
  case FDRState::Token::NEW_CPU_ID_RECORD:
    return "NEW_CPU_ID_RECORD";
  case FDRState::Token::FUNCTION_SEQUENCE:
    return "FUNCTION_SEQUENCE";
  case FDRState::Token::SCAN_TO_END_OF_THREAD_BUF:
    return "SCAN_TO_END_OF_THREAD_BUF";
  case FDRState::Token::CUSTOM_EVENT_DATA:
    return "CUSTOM_EVENT_DATA";
  case FDRState::Token::CALL_ARGUMENT:
    return "CALL_ARGUMENT";
  case FDRState::Token::BUFFER_EXTENTS:
    return "BUFFER_EXTENTS";
  case FDRState::Token::PID_RECORD:
    return "PID_RECORD";
  }
  return "UNKNOWN";
}

/// State transition when a NewBufferRecord is encountered.
Error processFDRNewBufferRecord(FDRState &State, DataExtractor &RecordExtractor,
                                uint32_t &OffsetPtr) {
  if (State.Expects != FDRState::Token::NEW_BUFFER_RECORD_OR_EOF)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Malformed log: Read New Buffer record kind out of sequence; expected: "
        "%s at offset %d.",
        fdrStateToTwine(State.Expects), OffsetPtr);

  auto PreReadOffset = OffsetPtr;
  State.ThreadId = RecordExtractor.getU16(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed reading the thread id at offset %d.", OffsetPtr);
  State.Expects = FDRState::Token::WALLCLOCK_RECORD;

  // Advance the offset pointer by enough bytes representing the remaining
  // padding in a metadata record.
  OffsetPtr += kFDRMetadataBodySize - 2;
  assert(OffsetPtr - PreReadOffset == kFDRMetadataBodySize);
  return Error::success();
}

/// State transition when an EndOfBufferRecord is encountered.
Error processFDREndOfBufferRecord(FDRState &State, uint32_t &OffsetPtr) {
  if (State.Expects == FDRState::Token::NEW_BUFFER_RECORD_OR_EOF)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Malformed log: Received EOB message without current buffer; expected: "
        "%s at offset %d.",
        fdrStateToTwine(State.Expects), OffsetPtr);

  State.Expects = FDRState::Token::SCAN_TO_END_OF_THREAD_BUF;

  // Advance the offset pointer by enough bytes representing the remaining
  // padding in a metadata record.
  OffsetPtr += kFDRMetadataBodySize;
  return Error::success();
}

/// State transition when a NewCPUIdRecord is encountered.
Error processFDRNewCPUIdRecord(FDRState &State, DataExtractor &RecordExtractor,
                               uint32_t &OffsetPtr) {
  if (State.Expects != FDRState::Token::FUNCTION_SEQUENCE &&
      State.Expects != FDRState::Token::NEW_CPU_ID_RECORD)
    return make_error<StringError>(
        Twine("Malformed log. Read NewCPUId record kind out of sequence; "
              "expected: ") +
            fdrStateToTwine(State.Expects),
        std::make_error_code(std::errc::executable_format_error));
  auto BeginOffset = OffsetPtr;
  auto PreReadOffset = OffsetPtr;
  State.CPUId = RecordExtractor.getU16(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed reading the CPU field at offset %d.", OffsetPtr);

  PreReadOffset = OffsetPtr;
  State.BaseTSC = RecordExtractor.getU64(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed reading the base TSC field at offset %d.", OffsetPtr);

  State.Expects = FDRState::Token::FUNCTION_SEQUENCE;

  // Advance the offset pointer by a few bytes, to account for the padding in
  // CPU ID metadata records that we've already advanced through.
  OffsetPtr += kFDRMetadataBodySize - (OffsetPtr - BeginOffset);
  assert(OffsetPtr - BeginOffset == kFDRMetadataBodySize);
  return Error::success();
}

/// State transition when a TSCWrapRecord (overflow detection) is encountered.
Error processFDRTSCWrapRecord(FDRState &State, DataExtractor &RecordExtractor,
                              uint32_t &OffsetPtr) {
  if (State.Expects != FDRState::Token::FUNCTION_SEQUENCE)
    return make_error<StringError>(
        Twine("Malformed log. Read TSCWrap record kind out of sequence; "
              "expecting: ") +
            fdrStateToTwine(State.Expects),
        std::make_error_code(std::errc::executable_format_error));
  auto PreReadOffset = OffsetPtr;
  State.BaseTSC = RecordExtractor.getU64(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed reading the base TSC field at offset %d.", OffsetPtr);

  // Advance the offset pointer by a few more bytes, accounting for the padding
  // in the metadata record after reading the base TSC.
  OffsetPtr += kFDRMetadataBodySize - 8;
  assert(OffsetPtr - PreReadOffset == kFDRMetadataBodySize);
  return Error::success();
}

/// State transition when a WallTimeMarkerRecord is encountered.
Error processFDRWallTimeRecord(FDRState &State, DataExtractor &RecordExtractor,
                               uint32_t &OffsetPtr) {
  if (State.Expects != FDRState::Token::WALLCLOCK_RECORD)
    return make_error<StringError>(
        Twine("Malformed log. Read Wallclock record kind out of sequence; "
              "expecting: ") +
            fdrStateToTwine(State.Expects),
        std::make_error_code(std::errc::executable_format_error));

  // Read in the data from the walltime record.
  auto PreReadOffset = OffsetPtr;
  auto WallTime = RecordExtractor.getU64(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed reading the walltime record at offset %d.", OffsetPtr);

  // TODO: Someday, reconcile the TSC ticks to wall clock time for presentation
  // purposes. For now, we're ignoring these records.
  (void)WallTime;
  State.Expects = FDRState::Token::NEW_CPU_ID_RECORD;

  // Advance the offset pointer by a few more bytes, accounting for the padding
  // in the metadata record after reading in the walltime data.
  OffsetPtr += kFDRMetadataBodySize - 8;
  assert(OffsetPtr - PreReadOffset == kFDRMetadataBodySize);
  return Error::success();
}

/// State transition when a PidRecord is encountered.
Error processFDRPidRecord(FDRState &State, DataExtractor &RecordExtractor,
                          uint32_t &OffsetPtr) {
  if (State.Expects != FDRState::Token::PID_RECORD)
    return make_error<StringError>(
        Twine("Malformed log. Read Pid record kind out of sequence; "
              "expected: ") +
            fdrStateToTwine(State.Expects),
        std::make_error_code(std::errc::executable_format_error));
  auto PreReadOffset = OffsetPtr;
  State.ProcessId = RecordExtractor.getU32(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed reading the process ID at offset %d.", OffsetPtr);
  State.Expects = FDRState::Token::NEW_CPU_ID_RECORD;

  // Advance the offset pointer by a few more bytes, accounting for the padding
  // in the metadata record after reading in the PID.
  OffsetPtr += kFDRMetadataBodySize - 4;
  assert(OffsetPtr - PreReadOffset == kFDRMetadataBodySize);
  return Error::success();
}

/// State transition when a CustomEventMarker is encountered.
Error processCustomEventMarker(FDRState &State, DataExtractor &RecordExtractor,
                               uint32_t &OffsetPtr) {
  // We can encounter a CustomEventMarker anywhere in the log, so we can handle
  // it regardless of the expectation. However, we do set the expectation to
  // read a set number of fixed bytes, as described in the metadata.
  auto BeginOffset = OffsetPtr;
  auto PreReadOffset = OffsetPtr;
  uint32_t DataSize = RecordExtractor.getU32(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed reading a custom event marker at offset %d.", OffsetPtr);

  PreReadOffset = OffsetPtr;
  uint64_t TSC = RecordExtractor.getU64(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed reading the TSC at offset %d.", OffsetPtr);

  // FIXME: Actually represent the record through the API. For now we only
  // skip through the data.
  (void)TSC;
  // Advance the offset ptr by the size of the data associated with the custom
  // event, as well as the padding associated with the remainder of the metadata
  // record.
  OffsetPtr += (kFDRMetadataBodySize - (OffsetPtr - BeginOffset)) + DataSize;
  if (!RecordExtractor.isValidOffset(OffsetPtr))
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Reading custom event data moves past addressable trace data (starting "
        "at offset %d, advancing to offset %d).",
        BeginOffset, OffsetPtr);
  return Error::success();
}

/// State transition when an BufferExtents record is encountered.
Error processBufferExtents(FDRState &State, DataExtractor &RecordExtractor,
                           uint32_t &OffsetPtr) {
  if (State.Expects != FDRState::Token::BUFFER_EXTENTS)
    return make_error<StringError>(
        Twine("Malformed log. Buffer Extents unexpected; expected: ") +
            fdrStateToTwine(State.Expects),
        std::make_error_code(std::errc::executable_format_error));

  auto PreReadOffset = OffsetPtr;
  State.CurrentBufferSize = RecordExtractor.getU64(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed to read current buffer size at offset %d.", OffsetPtr);

  State.Expects = FDRState::Token::NEW_BUFFER_RECORD_OR_EOF;

  // Advance the offset pointer by enough bytes accounting for the padding in a
  // metadata record, after we read in the buffer extents.
  OffsetPtr += kFDRMetadataBodySize - 8;
  return Error::success();
}

/// State transition when a CallArgumentRecord is encountered.
Error processFDRCallArgumentRecord(FDRState &State,
                                   DataExtractor &RecordExtractor,
                                   std::vector<XRayRecord> &Records,
                                   uint32_t &OffsetPtr) {
  auto &Enter = Records.back();
  if (Enter.Type != RecordTypes::ENTER && Enter.Type != RecordTypes::ENTER_ARG)
    return make_error<StringError>(
        "CallArgument needs to be right after a function entry",
        std::make_error_code(std::errc::executable_format_error));

  auto PreReadOffset = OffsetPtr;
  auto Arg = RecordExtractor.getU64(&OffsetPtr);
  if (OffsetPtr == PreReadOffset)
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Failed to read argument record at offset %d.", OffsetPtr);

  Enter.Type = RecordTypes::ENTER_ARG;
  Enter.CallArgs.emplace_back(Arg);

  // Advance the offset pointer by enough bytes accounting for the padding in a
  // metadata record, after reading the payload.
  OffsetPtr += kFDRMetadataBodySize - 8;
  return Error::success();
}

/// Advances the state machine for reading the FDR record type by reading one
/// Metadata Record and updating the State appropriately based on the kind of
/// record encountered. The RecordKind is encoded in the first byte of the
/// Record, which the caller should pass in because they have already read it
/// to determine that this is a metadata record as opposed to a function record.
///
/// Beginning with Version 2 of the FDR log, we do not depend on the size of the
/// buffer, but rather use the extents to determine how far to read in the log
/// for this particular buffer.
///
/// In Version 3, FDR log now includes a pid metadata record after
/// WallTimeMarker
Error processFDRMetadataRecord(FDRState &State, DataExtractor &RecordExtractor,
                               uint32_t &OffsetPtr,
                               std::vector<XRayRecord> &Records,
                               uint16_t Version, uint8_t FirstByte) {
  // The remaining 7 bits of the first byte are the RecordKind enum for each
  // Metadata Record.
  switch (FirstByte >> 1) {
  case 0: // NewBuffer
    if (auto E = processFDRNewBufferRecord(State, RecordExtractor, OffsetPtr))
      return E;
    break;
  case 1: // EndOfBuffer
    if (Version >= 2)
      return make_error<StringError>(
          "Since Version 2 of FDR logging, we no longer support EOB records.",
          std::make_error_code(std::errc::executable_format_error));
    if (auto E = processFDREndOfBufferRecord(State, OffsetPtr))
      return E;
    break;
  case 2: // NewCPUId
    if (auto E = processFDRNewCPUIdRecord(State, RecordExtractor, OffsetPtr))
      return E;
    break;
  case 3: // TSCWrap
    if (auto E = processFDRTSCWrapRecord(State, RecordExtractor, OffsetPtr))
      return E;
    break;
  case 4: // WallTimeMarker
    if (auto E = processFDRWallTimeRecord(State, RecordExtractor, OffsetPtr))
      return E;
    // In Version 3 and and above, a PidRecord is expected after WallTimeRecord
    if (Version >= 3)
      State.Expects = FDRState::Token::PID_RECORD;
    break;
  case 5: // CustomEventMarker
    if (auto E = processCustomEventMarker(State, RecordExtractor, OffsetPtr))
      return E;
    break;
  case 6: // CallArgument
    if (auto E = processFDRCallArgumentRecord(State, RecordExtractor, Records,
                                              OffsetPtr))
      return E;
    break;
  case 7: // BufferExtents
    if (auto E = processBufferExtents(State, RecordExtractor, OffsetPtr))
      return E;
    break;
  case 9: // Pid
    if (auto E = processFDRPidRecord(State, RecordExtractor, OffsetPtr))
      return E;
    break;
  default:
    return createStringError(
        std::make_error_code(std::errc::executable_format_error),
        "Illegal metadata record type: '%d' at offset %d.", FirstByte >> 1,
        OffsetPtr);
  }
  return Error::success();
}

/// Reads a function record from an FDR format log, appending a new XRayRecord
/// to the vector being populated and updating the State with a new value
/// reference value to interpret TSC deltas.
///
/// The XRayRecord constructed includes information from the function record
/// processed here as well as Thread ID and CPU ID formerly extracted into
/// State.
Error processFDRFunctionRecord(FDRState &State, DataExtractor &RecordExtractor,
                               uint32_t &OffsetPtr, uint8_t FirstByte,
                               std::vector<XRayRecord> &Records) {
  switch (State.Expects) {
  case FDRState::Token::NEW_BUFFER_RECORD_OR_EOF:
    return make_error<StringError>(
        "Malformed log. Received Function Record before new buffer setup.",
        std::make_error_code(std::errc::executable_format_error));
  case FDRState::Token::WALLCLOCK_RECORD:
    return make_error<StringError>(
        "Malformed log. Received Function Record when expecting wallclock.",
        std::make_error_code(std::errc::executable_format_error));
  case FDRState::Token::PID_RECORD:
    return make_error<StringError>(
        "Malformed log. Received Function Record when expecting pid.",
        std::make_error_code(std::errc::executable_format_error));
  case FDRState::Token::NEW_CPU_ID_RECORD:
    return make_error<StringError>(
        "Malformed log. Received Function Record before first CPU record.",
        std::make_error_code(std::errc::executable_format_error));
  default:
    Records.emplace_back();
    auto &Record = Records.back();
    Record.RecordType = 0; // Record is type NORMAL.
    // Strip off record type bit and use the next three bits.
    auto T = (FirstByte >> 1) & 0x07;
    switch (T) {
    case static_cast<decltype(T)>(RecordTypes::ENTER):
      Record.Type = RecordTypes::ENTER;
      break;
    case static_cast<decltype(T)>(RecordTypes::EXIT):
      Record.Type = RecordTypes::EXIT;
      break;
    case static_cast<decltype(T)>(RecordTypes::TAIL_EXIT):
      Record.Type = RecordTypes::TAIL_EXIT;
      break;
    case static_cast<decltype(T)>(RecordTypes::ENTER_ARG):
      Record.Type = RecordTypes::ENTER_ARG;
      State.Expects = FDRState::Token::CALL_ARGUMENT;
      break;
    default:
      return createStringError(
          std::make_error_code(std::errc::executable_format_error),
          "Illegal function record type '%d' at offset %d.", T, OffsetPtr);
    }
    Record.CPU = State.CPUId;
    Record.TId = State.ThreadId;
    Record.PId = State.ProcessId;

    // Back up one byte to re-read the first byte, which is important for
    // computing the function id for a record.
    --OffsetPtr;

    // Despite function Id being a signed int on XRayRecord,
    // when it is written to an FDR format, the top bits are truncated,
    // so it is effectively an unsigned value. When we shift off the
    // top four bits, we want the shift to be logical, so we read as
    // uint32_t.
    auto PreReadOffset = OffsetPtr;
    uint32_t FuncIdBitField = RecordExtractor.getU32(&OffsetPtr);
    if (OffsetPtr == PreReadOffset)
      return createStringError(
          std::make_error_code(std::errc::executable_format_error),
          "Failed reading truncated function id field at offset %d.",
          OffsetPtr);

    Record.FuncId = FuncIdBitField >> 4;

    // FunctionRecords have a 32 bit delta from the previous absolute TSC
    // or TSC delta. If this would overflow, we should read a TSCWrap record
    // with an absolute TSC reading.
    PreReadOffset = OffsetPtr;
    uint64_t NewTSC = State.BaseTSC + RecordExtractor.getU32(&OffsetPtr);
    if (OffsetPtr == PreReadOffset)
      return createStringError(
          std::make_error_code(std::errc::executable_format_error),
          "Failed reading TSC delta at offset %d.", OffsetPtr);

    State.BaseTSC = NewTSC;
    Record.TSC = NewTSC;
  }
  return Error::success();
}

/// Reads a log in FDR mode for version 1 of this binary format. FDR mode is
/// defined as part of the compiler-rt project in xray_fdr_logging.h, and such
/// a log consists of the familiar 32 bit XRayHeader, followed by sequences of
/// of interspersed 16 byte Metadata Records and 8 byte Function Records.
///
/// The following is an attempt to document the grammar of the format, which is
/// parsed by this function for little-endian machines. Since the format makes
/// use of BitFields, when we support big-endian architectures, we will need to
/// adjust not only the endianness parameter to llvm's RecordExtractor, but also
/// the bit twiddling logic, which is consistent with the little-endian
/// convention that BitFields within a struct will first be packed into the
/// least significant bits the address they belong to.
///
/// We expect a format complying with the grammar in the following pseudo-EBNF
/// in Version 1 of the FDR log.
///
/// FDRLog: XRayFileHeader ThreadBuffer*
/// XRayFileHeader: 32 bytes to identify the log as FDR with machine metadata.
///     Includes BufferSize
/// ThreadBuffer: NewBuffer WallClockTime NewCPUId FunctionSequence EOB
/// BufSize: 8 byte unsigned integer indicating how large the buffer is.
/// NewBuffer: 16 byte metadata record with Thread Id.
/// WallClockTime: 16 byte metadata record with human readable time.
/// Pid: 16 byte metadata record with Pid
/// NewCPUId: 16 byte metadata record with CPUId and a 64 bit TSC reading.
/// EOB: 16 byte record in a thread buffer plus mem garbage to fill BufSize.
/// FunctionSequence: NewCPUId | TSCWrap | FunctionRecord
/// TSCWrap: 16 byte metadata record with a full 64 bit TSC reading.
/// FunctionRecord: 8 byte record with FunctionId, entry/exit, and TSC delta.
///
/// In Version 2, we make the following changes:
///
/// ThreadBuffer: BufferExtents NewBuffer WallClockTime NewCPUId
///               FunctionSequence
/// BufferExtents: 16 byte metdata record describing how many usable bytes are
///                in the buffer. This is measured from the start of the buffer
///                and must always be at least 48 (bytes).
///
/// In Version 3, we make the following changes:
///
/// ThreadBuffer: BufferExtents NewBuffer WallClockTime Pid NewCPUId
///               FunctionSequence
/// EOB: *deprecated*
Error loadFDRLog(StringRef Data, XRayFileHeader &FileHeader,
                 std::vector<XRayRecord> &Records) {

  if (Data.size() < 32)
    return make_error<StringError>(
        "Not enough bytes for an XRay log.",
        std::make_error_code(std::errc::invalid_argument));

  DataExtractor Reader(Data, true, 8);
  uint32_t OffsetPtr = 0;

  auto FileHeaderOrError = readBinaryFormatHeader(Reader, OffsetPtr);
  if (!FileHeaderOrError)
    return FileHeaderOrError.takeError();
  FileHeader = std::move(FileHeaderOrError.get());

  uint64_t BufferSize = 0;
  {
    StringRef ExtraDataRef(FileHeader.FreeFormData, 16);
    DataExtractor ExtraDataExtractor(ExtraDataRef, true, 8);
    uint32_t ExtraDataOffset = 0;
    BufferSize = ExtraDataExtractor.getU64(&ExtraDataOffset);
  }

  FDRState::Token InitialExpectation;
  switch (FileHeader.Version) {
  case 1:
    InitialExpectation = FDRState::Token::NEW_BUFFER_RECORD_OR_EOF;
    break;
  case 2:
  case 3:
    InitialExpectation = FDRState::Token::BUFFER_EXTENTS;
    break;
  default:
    return make_error<StringError>(
        Twine("Unsupported version '") + Twine(FileHeader.Version) + "'",
        std::make_error_code(std::errc::executable_format_error));
  }
  FDRState State{0, 0, 0, 0, InitialExpectation, BufferSize, 0};

  // RecordSize will tell the loop how far to seek ahead based on the record
  // type that we have just read.
  while (Reader.isValidOffset(OffsetPtr)) {
    auto BeginOffset = OffsetPtr;
    if (State.Expects == FDRState::Token::SCAN_TO_END_OF_THREAD_BUF) {
      OffsetPtr += State.CurrentBufferSize - State.CurrentBufferConsumed;
      State.CurrentBufferConsumed = 0;
      State.Expects = FDRState::Token::NEW_BUFFER_RECORD_OR_EOF;
      continue;
    }
    auto PreReadOffset = OffsetPtr;
    uint8_t BitField = Reader.getU8(&OffsetPtr);
    if (OffsetPtr == PreReadOffset)
      return createStringError(
          std::make_error_code(std::errc::executable_format_error),
          "Failed reading first byte of record at offset %d.", OffsetPtr);
    bool isMetadataRecord = BitField & 0x01uL;
    bool isBufferExtents =
        (BitField >> 1) == 7; // BufferExtents record kind == 7
    if (isMetadataRecord) {
      if (auto E = processFDRMetadataRecord(State, Reader, OffsetPtr, Records,
                                            FileHeader.Version, BitField))
        return E;
    } else { // Process Function Record
      if (auto E = processFDRFunctionRecord(State, Reader, OffsetPtr, BitField,
                                            Records))
        return E;
    }

    // The BufferExtents record is technically not part of the buffer, so we
    // don't count the size of that record against the buffer's actual size.
    if (!isBufferExtents)
      State.CurrentBufferConsumed += OffsetPtr - BeginOffset;

    assert(State.CurrentBufferConsumed <= State.CurrentBufferSize);

    if ((FileHeader.Version == 2 || FileHeader.Version == 3) &&
        State.CurrentBufferSize == State.CurrentBufferConsumed) {
      // In Version 2 of the log, we don't need to scan to the end of the thread
      // buffer if we've already consumed all the bytes we need to.
      State.Expects = FDRState::Token::BUFFER_EXTENTS;
      State.CurrentBufferSize = BufferSize;
      State.CurrentBufferConsumed = 0;
    }
  }

  // Having iterated over everything we've been given, we've either consumed
  // everything and ended up in the end state, or were told to skip the rest.
  bool Finished = State.Expects == FDRState::Token::SCAN_TO_END_OF_THREAD_BUF &&
                  State.CurrentBufferSize == State.CurrentBufferConsumed;
  if ((State.Expects != FDRState::Token::NEW_BUFFER_RECORD_OR_EOF &&
       State.Expects != FDRState::Token::BUFFER_EXTENTS) &&
      !Finished)
    return make_error<StringError>(
        Twine("Encountered EOF with unexpected state expectation ") +
            fdrStateToTwine(State.Expects) +
            ". Remaining expected bytes in thread buffer total " +
            Twine(State.CurrentBufferSize - State.CurrentBufferConsumed),
        std::make_error_code(std::errc::executable_format_error));

  return Error::success();
}

Error loadYAMLLog(StringRef Data, XRayFileHeader &FileHeader,
                  std::vector<XRayRecord> &Records) {
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
                   return XRayRecord{R.RecordType, R.CPU, R.Type, R.FuncId,
                                     R.TSC,        R.TId, R.PId,  R.CallArgs};
                 });
  return Error::success();
}
} // namespace

Expected<Trace> llvm::xray::loadTraceFile(StringRef Filename, bool Sort) {
  int Fd;
  if (auto EC = sys::fs::openFileForRead(Filename, Fd)) {
    return make_error<StringError>(
        Twine("Cannot read log from '") + Filename + "'", EC);
  }

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

  // Map the opened file into memory and use a StringRef to access it later.
  std::error_code EC;
  sys::fs::mapped_file_region MappedFile(
      Fd, sys::fs::mapped_file_region::mapmode::readonly, FileSize, 0, EC);
  if (EC) {
    return make_error<StringError>(
        Twine("Cannot read log from '") + Filename + "'", EC);
  }
  auto Data = StringRef(MappedFile.data(), MappedFile.size());

  // Attempt to detect the file type using file magic. We have a slight bias
  // towards the binary format, and we do this by making sure that the first 4
  // bytes of the binary file is some combination of the following byte
  // patterns: (observe the code loading them assumes they're little endian)
  //
  //   0x01 0x00 0x00 0x00 - version 1, "naive" format
  //   0x01 0x00 0x01 0x00 - version 1, "flight data recorder" format
  //   0x02 0x00 0x01 0x00 - version 2, "flight data recorder" format
  //
  // YAML files don't typically have those first four bytes as valid text so we
  // try loading assuming YAML if we don't find these bytes.
  //
  // Only if we can't load either the binary or the YAML format will we yield an
  // error.
  StringRef Magic(MappedFile.data(), 4);
  DataExtractor HeaderExtractor(Magic, true, 8);
  uint32_t OffsetPtr = 0;
  uint16_t Version = HeaderExtractor.getU16(&OffsetPtr);
  uint16_t Type = HeaderExtractor.getU16(&OffsetPtr);

  enum BinaryFormatType { NAIVE_FORMAT = 0, FLIGHT_DATA_RECORDER_FORMAT = 1 };

  Trace T;
  switch (Type) {
  case NAIVE_FORMAT:
    if (Version == 1 || Version == 2 || Version == 3) {
      if (auto E = loadNaiveFormatLog(Data, T.FileHeader, T.Records))
        return std::move(E);
    } else {
      return make_error<StringError>(
          Twine("Unsupported version for Basic/Naive Mode logging: ") +
              Twine(Version),
          std::make_error_code(std::errc::executable_format_error));
    }
    break;
  case FLIGHT_DATA_RECORDER_FORMAT:
    if (Version == 1 || Version == 2 || Version == 3) {
      if (auto E = loadFDRLog(Data, T.FileHeader, T.Records))
        return std::move(E);
    } else {
      return make_error<StringError>(
          Twine("Unsupported version for FDR Mode logging: ") + Twine(Version),
          std::make_error_code(std::errc::executable_format_error));
    }
    break;
  default:
    if (auto E = loadYAMLLog(Data, T.FileHeader, T.Records))
      return std::move(E);
  }

  if (Sort)
    std::stable_sort(T.Records.begin(), T.Records.end(),
                     [&](const XRayRecord &L, const XRayRecord &R) {
                       return L.TSC < R.TSC;
                     });

  return std::move(T);
}
