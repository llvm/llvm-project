#include "llvm/DebugInfo/PDB/Raw/TpiStreamBuilder.h"

#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
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

TpiStreamBuilder::TpiStreamBuilder(BumpPtrAllocator &Allocator)
    : Allocator(Allocator), Header(nullptr) {}

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

  H->Version = *VerHeader;
  H->HeaderSize = sizeof(TpiStreamHeader);
  H->TypeIndexBegin = codeview::TypeIndex::FirstNonSimpleIndex;
  H->TypeIndexEnd = H->TypeIndexBegin + Count;
  H->TypeRecordBytes = TypeRecordStream.getLength();

  H->HashStreamIndex = kInvalidStreamIndex;
  H->HashAuxStreamIndex = kInvalidStreamIndex;
  H->HashKeySize = sizeof(ulittle32_t);
  H->NumHashBuckets = MinTpiHashBuckets;

  H->HashValueBuffer.Length = 0;
  H->HashAdjBuffer.Length = 0;
  H->IndexOffsetBuffer.Length = 0;

  Header = H;
  return Error::success();
}

uint32_t TpiStreamBuilder::calculateSerializedLength() const {
  return sizeof(TpiStreamHeader) + TypeRecordStream.getLength();
}

Expected<std::unique_ptr<TpiStream>>
TpiStreamBuilder::build(PDBFile &File, const msf::WritableStream &Buffer) {
  if (!VerHeader.hasValue())
    return make_error<RawError>(raw_error_code::unspecified,
                                "Missing TPI Stream Version");
  if (auto EC = finalize())
    return std::move(EC);

  auto StreamData = MappedBlockStream::createIndexedStream(File.getMsfLayout(),
                                                           Buffer, StreamTPI);
  auto Tpi = llvm::make_unique<TpiStream>(File, std::move(StreamData));
  Tpi->Header = Header;
  Tpi->TypeRecords = VarStreamArray<codeview::CVType>(TypeRecordStream);
  return std::move(Tpi);
}

Error TpiStreamBuilder::commit(const msf::MSFLayout &Layout,
                               const msf::WritableStream &Buffer) {
  if (auto EC = finalize())
    return EC;

  auto InfoS =
      WritableMappedBlockStream::createIndexedStream(Layout, Buffer, StreamTPI);

  StreamWriter Writer(*InfoS);
  if (auto EC = Writer.writeObject(*Header))
    return EC;

  auto RecordArray = VarStreamArray<codeview::CVType>(TypeRecordStream);
  if (auto EC = Writer.writeArray(RecordArray))
    return EC;

  return Error::success();
}
