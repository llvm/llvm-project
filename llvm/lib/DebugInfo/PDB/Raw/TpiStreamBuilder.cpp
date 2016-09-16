#include "llvm/DebugInfo/PDB/Raw/TpiStreamBuilder.h"

#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/MSF/MSFBuilder.h"
#include "llvm/DebugInfo/MSF/MappedBlockStream.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/TpiStream.h"
#include "llvm/Support/Allocator.h"

using namespace llvm;
using namespace llvm::msf;
using namespace llvm::pdb;
using namespace llvm::support;

TpiStreamBuilder::TpiStreamBuilder(MSFBuilder &Msf, uint32_t StreamIdx)
    : Msf(Msf), Allocator(Msf.getAllocator()), Header(nullptr), Idx(StreamIdx) {
}

TpiStreamBuilder::~TpiStreamBuilder() {}

void TpiStreamBuilder::setVersionHeader(PdbRaw_TpiVer Version) {
  VerHeader = Version;
}

void TpiStreamBuilder::addTypeRecord(const codeview::CVType &Record) {
  TypeRecords.push_back(Record);
  TypeRecordStream.setItems(TypeRecords);
}

Error TpiStreamBuilder::finalize() {
  if (Header)
    return Error::success();

  TpiStreamHeader *H = Allocator.Allocate<TpiStreamHeader>();

  uint32_t Count = TypeRecords.size();
  uint32_t HashBufferSize = calculateHashBufferSize();

  H->Version = *VerHeader;
  H->HeaderSize = sizeof(TpiStreamHeader);
  H->TypeIndexBegin = codeview::TypeIndex::FirstNonSimpleIndex;
  H->TypeIndexEnd = H->TypeIndexBegin + Count;
  H->TypeRecordBytes = TypeRecordStream.getLength();

  H->HashStreamIndex = HashStreamIndex;
  H->HashAuxStreamIndex = kInvalidStreamIndex;
  H->HashKeySize = sizeof(ulittle32_t);
  H->NumHashBuckets = MinTpiHashBuckets;

  // Recall that hash values go into a completely different stream identified by
  // the `HashStreamIndex` field of the `TpiStreamHeader`.  Therefore, the data
  // begins at offset 0 of this independent stream.
  H->HashValueBuffer.Off = 0;
  H->HashValueBuffer.Length = HashBufferSize;
  H->HashAdjBuffer.Off = H->HashValueBuffer.Off + H->HashValueBuffer.Length;
  H->HashAdjBuffer.Length = 0;
  H->IndexOffsetBuffer.Off = H->HashAdjBuffer.Off + H->HashAdjBuffer.Length;
  H->IndexOffsetBuffer.Length = 0;

  Header = H;
  return Error::success();
}

uint32_t TpiStreamBuilder::calculateSerializedLength() const {
  return sizeof(TpiStreamHeader) + TypeRecordStream.getLength();
}

uint32_t TpiStreamBuilder::calculateHashBufferSize() const {
  if (TypeRecords.empty() || !TypeRecords[0].Hash.hasValue())
    return 0;
  return TypeRecords.size() * sizeof(ulittle32_t);
}

Error TpiStreamBuilder::finalizeMsfLayout() {
  uint32_t Length = calculateSerializedLength();
  if (auto EC = Msf.setStreamSize(Idx, Length))
    return EC;

  uint32_t HashBufferSize = calculateHashBufferSize();

  if (HashBufferSize == 0)
    return Error::success();

  auto ExpectedIndex = Msf.addStream(HashBufferSize);
  if (!ExpectedIndex)
    return ExpectedIndex.takeError();
  HashStreamIndex = *ExpectedIndex;
  ulittle32_t *H = Allocator.Allocate<ulittle32_t>(TypeRecords.size());
  MutableArrayRef<ulittle32_t> HashBuffer(H, TypeRecords.size());
  for (uint32_t I = 0; I < TypeRecords.size(); ++I) {
    HashBuffer[I] = *TypeRecords[I].Hash % MinTpiHashBuckets;
  }
  ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(HashBuffer.data()),
                          HashBufferSize);
  HashValueStream = llvm::make_unique<ByteStream>(Bytes);
  return Error::success();
}

Expected<std::unique_ptr<TpiStream>>
TpiStreamBuilder::build(PDBFile &File, const msf::WritableStream &Buffer) {
  if (!VerHeader.hasValue())
    return make_error<RawError>(raw_error_code::unspecified,
                                "Missing TPI Stream Version");
  if (auto EC = finalize())
    return std::move(EC);

  auto StreamData =
      MappedBlockStream::createIndexedStream(File.getMsfLayout(), Buffer, Idx);
  auto Tpi = llvm::make_unique<TpiStream>(File, std::move(StreamData));
  Tpi->Header = Header;
  Tpi->TypeRecords = VarStreamArray<codeview::CVType>(TypeRecordStream);
  if (HashValueStream) {
    Tpi->HashStream = std::move(HashValueStream);
    StreamReader HSR(*Tpi->HashStream);
    if (auto EC = HSR.readArray(Tpi->HashValues, TypeRecords.size()))
      return std::move(EC);
  }
  return std::move(Tpi);
}

Error TpiStreamBuilder::commit(const msf::MSFLayout &Layout,
                               const msf::WritableStream &Buffer) {
  if (auto EC = finalize())
    return EC;

  auto InfoS =
      WritableMappedBlockStream::createIndexedStream(Layout, Buffer, Idx);

  StreamWriter Writer(*InfoS);
  if (auto EC = Writer.writeObject(*Header))
    return EC;

  auto RecordArray = VarStreamArray<codeview::CVType>(TypeRecordStream);
  if (auto EC = Writer.writeArray(RecordArray))
    return EC;

  if (HashStreamIndex != kInvalidStreamIndex) {
    auto HVS = WritableMappedBlockStream::createIndexedStream(Layout, Buffer,
                                                              HashStreamIndex);
    StreamWriter HW(*HVS);
    if (auto EC = HW.writeStreamRef(*HashValueStream))
      return EC;
  }

  return Error::success();
}
