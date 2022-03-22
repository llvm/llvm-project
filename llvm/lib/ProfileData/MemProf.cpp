#include "llvm/ProfileData/MemProf.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"

namespace llvm {
namespace memprof {

void MemProfRecord::serialize(const MemProfSchema &Schema, raw_ostream &OS) {
  using namespace support;

  endian::Writer LE(OS, little);

  LE.write<uint64_t>(AllocSites.size());
  for (const AllocationInfo &N : AllocSites) {
    LE.write<uint64_t>(N.CallStack.size());
    for (const Frame &F : N.CallStack)
      F.serialize(OS);
    N.Info.serialize(Schema, OS);
  }

  // Related contexts.
  LE.write<uint64_t>(CallSites.size());
  for (const auto &Frames : CallSites) {
    LE.write<uint64_t>(Frames.size());
    for (const Frame &F : Frames)
      F.serialize(OS);
  }
}

MemProfRecord MemProfRecord::deserialize(const MemProfSchema &Schema,
                                         const unsigned char *Ptr) {
  using namespace support;

  MemProfRecord Record;

  // Read the meminfo nodes.
  const uint64_t NumNodes = endian::readNext<uint64_t, little, unaligned>(Ptr);
  for (uint64_t I = 0; I < NumNodes; I++) {
    MemProfRecord::AllocationInfo Node;
    const uint64_t NumFrames =
        endian::readNext<uint64_t, little, unaligned>(Ptr);
    for (uint64_t J = 0; J < NumFrames; J++) {
      const auto F = MemProfRecord::Frame::deserialize(Ptr);
      Ptr += MemProfRecord::Frame::serializedSize();
      Node.CallStack.push_back(F);
    }
    Node.Info.deserialize(Schema, Ptr);
    Ptr += PortableMemInfoBlock::serializedSize();
    Record.AllocSites.push_back(Node);
  }

  // Read the callsite information.
  const uint64_t NumCtxs = endian::readNext<uint64_t, little, unaligned>(Ptr);
  for (uint64_t J = 0; J < NumCtxs; J++) {
    const uint64_t NumFrames =
        endian::readNext<uint64_t, little, unaligned>(Ptr);
    llvm::SmallVector<Frame> Frames;
    for (uint64_t K = 0; K < NumFrames; K++) {
      const auto F = MemProfRecord::Frame::deserialize(Ptr);
      Ptr += MemProfRecord::Frame::serializedSize();
      Frames.push_back(F);
    }
    Record.CallSites.push_back(Frames);
  }

  return Record;
}

GlobalValue::GUID MemProfRecord::getGUID(const StringRef FunctionName) {
  const auto Pos = FunctionName.find(".llvm.");

  // We use the function guid which we expect to be a uint64_t. At
  // this time, it is the lower 64 bits of the md5 of the function
  // name. Any suffix with .llvm. is trimmed since these are added by
  // thinLTO global promotion. At the time the profile is consumed,
  // these suffixes will not be present.
  return Function::getGUID(FunctionName.take_front(Pos));
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
