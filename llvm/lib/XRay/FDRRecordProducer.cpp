//===- FDRRecordProducer.cpp - XRay FDR Mode Record Producer --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/XRay/FDRRecordProducer.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {
namespace xray {

namespace {

// Keep this in sync with the values written in the XRay FDR mode runtime in
// compiler-rt.
enum class MetadataRecordKinds : uint8_t {
  NewBuffer,
  EndOfBuffer,
  NewCPUId,
  TSCWrap,
  WalltimeMarker,
  CustomEventMarker,
  CallArgument,
  BufferExtents,
  TypedEventMarker,
  Pid,
  // This is an end marker, used to identify the upper bound for this enum.
  EnumEndMarker,
};

Expected<std::unique_ptr<Record>>
metadataRecordType(const XRayFileHeader &Header, uint8_t T) {
  if (T >= static_cast<uint8_t>(MetadataRecordKinds::EnumEndMarker))
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Invalid metadata record type: %d", T);
  static constexpr MetadataRecordKinds Mapping[] = {
      MetadataRecordKinds::NewBuffer,
      MetadataRecordKinds::EndOfBuffer,
      MetadataRecordKinds::NewCPUId,
      MetadataRecordKinds::TSCWrap,
      MetadataRecordKinds::WalltimeMarker,
      MetadataRecordKinds::CustomEventMarker,
      MetadataRecordKinds::CallArgument,
      MetadataRecordKinds::BufferExtents,
      MetadataRecordKinds::TypedEventMarker,
      MetadataRecordKinds::Pid,
  };
  switch (Mapping[T]) {
  case MetadataRecordKinds::NewBuffer:
    return make_unique<NewBufferRecord>();
  case MetadataRecordKinds::EndOfBuffer:
    if (Header.Version >= 2)
      return createStringError(
          std::make_error_code(std::errc::executable_format_error),
          "End of buffer records are no longer supported starting version "
          "2 of the log.");
    return make_unique<EndBufferRecord>();
  case MetadataRecordKinds::NewCPUId:
    return make_unique<NewCPUIDRecord>();
  case MetadataRecordKinds::TSCWrap:
    return make_unique<TSCWrapRecord>();
  case MetadataRecordKinds::WalltimeMarker:
    return make_unique<WallclockRecord>();
  case MetadataRecordKinds::CustomEventMarker:
    return make_unique<CustomEventRecord>();
  case MetadataRecordKinds::CallArgument:
    return make_unique<CallArgRecord>();
  case MetadataRecordKinds::BufferExtents:
    return make_unique<BufferExtents>();
  case MetadataRecordKinds::TypedEventMarker:
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Encountered an unsupported TypedEventMarker.");
  case MetadataRecordKinds::Pid:
    return make_unique<PIDRecord>();
  case MetadataRecordKinds::EnumEndMarker:
    llvm_unreachable("Invalid MetadataRecordKind");
  }
  llvm_unreachable("Unhandled MetadataRecordKinds enum value");
}

} // namespace

Expected<std::unique_ptr<Record>> FileBasedRecordProducer::produce() {
  // At the top level, we read one byte to determine the type of the record to
  // create. This byte will comprise of the following bits:
  //
  //   - offset 0: A '1' indicates a metadata record, a '0' indicates a function
  //     record.
  //   - offsets 1-7: For metadata records, this will indicate the kind of
  //     metadata record should be loaded.
  //
  // We read first byte, then create the appropriate type of record to consume
  // the rest of the bytes.
  auto PreReadOffset = OffsetPtr;
  uint8_t FirstByte = E.getU8(&OffsetPtr);
  std::unique_ptr<Record> R;

  // For metadata records, handle especially here.
  if (FirstByte & 0x01) {
    auto LoadedType = FirstByte >> 1;
    auto MetadataRecordOrErr = metadataRecordType(Header, LoadedType);
    if (!MetadataRecordOrErr)
      return joinErrors(
          MetadataRecordOrErr.takeError(),
          createStringError(
              std::make_error_code(std::errc::executable_format_error),
              "Encountered an unsupported metadata record (%d) at offset %d.",
              LoadedType, PreReadOffset));
    R = std::move(MetadataRecordOrErr.get());
  } else {
    R = llvm::make_unique<FunctionRecord>();
  }
  RecordInitializer RI(E, OffsetPtr);

  if (auto Err = R->apply(RI))
    return std::move(Err);

  assert(R != nullptr);
  return std::move(R);
}

} // namespace xray
} // namespace llvm
