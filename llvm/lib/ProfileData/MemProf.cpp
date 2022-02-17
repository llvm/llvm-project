#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"

namespace llvm {
namespace memprof {

void serializeRecords(const ArrayRef<MemProfRecord> Records,
                      const MemProfSchema &Schema, raw_ostream &OS) {
  using namespace support;

  endian::Writer LE(OS, little);

  LE.write<uint64_t>(Records.size());
  for (const MemProfRecord &MR : Records) {
    LE.write<uint64_t>(MR.CallStack.size());
    for (const MemProfRecord::Frame &F : MR.CallStack) {
      F.serialize(OS);
    }
    MR.Info.serialize(Schema, OS);
  }
}

SmallVector<MemProfRecord, 4> deserializeRecords(const MemProfSchema &Schema,
                                                 const unsigned char *Ptr) {
  using namespace support;

  SmallVector<MemProfRecord, 4> Records;
  const uint64_t NumRecords =
      endian::readNext<uint64_t, little, unaligned>(Ptr);
  for (uint64_t I = 0; I < NumRecords; I++) {
    MemProfRecord MR;
    const uint64_t NumFrames =
        endian::readNext<uint64_t, little, unaligned>(Ptr);
    for (uint64_t J = 0; J < NumFrames; J++) {
      const auto F = MemProfRecord::Frame::deserialize(Ptr);
      Ptr += MemProfRecord::Frame::serializedSize();
      MR.CallStack.push_back(F);
    }
    MR.Info.deserialize(Schema, Ptr);
    Ptr += PortableMemInfoBlock::serializedSize();
    Records.push_back(MR);
  }
  return Records;
}

Expected<MemProfSchema> readMemProfSchema(const unsigned char *&Buffer) {
  using namespace support;

  const unsigned char *Ptr = Buffer;
  const uint64_t NumSchemaIds =
      endian::readNext<uint64_t, little, unaligned>(Ptr);
  if (NumSchemaIds > static_cast<uint64_t>(Meta::Size)) {
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "memprof schema invalid");
  }

  MemProfSchema Result;
  for (size_t I = 0; I < NumSchemaIds; I++) {
    const uint64_t Tag = endian::readNext<uint64_t, little, unaligned>(Ptr);
    if (Tag >= static_cast<uint64_t>(Meta::Size)) {
      return make_error<InstrProfError>(instrprof_error::malformed,
                                        "memprof schema invalid");
    }
    Result.push_back(static_cast<Meta>(Tag));
  }
  // Advace the buffer to one past the schema if we succeeded.
  Buffer = Ptr;
  return Result;
}

} // namespace memprof
} // namespace llvm
