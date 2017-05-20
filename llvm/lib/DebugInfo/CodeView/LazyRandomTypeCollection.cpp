//===- LazyRandomTypeCollection.cpp ---------------------------- *- C++--*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"

#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/TypeDatabase.h"
#include "llvm/DebugInfo/CodeView/TypeServerHandler.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"

using namespace llvm;
using namespace llvm::codeview;

static void error(Error &&EC) {
  assert(!static_cast<bool>(EC));
  if (EC)
    consumeError(std::move(EC));
}

LazyRandomTypeCollection::LazyRandomTypeCollection(uint32_t RecordCountHint)
    : LazyRandomTypeCollection(CVTypeArray(), RecordCountHint,
                               PartialOffsetArray()) {}

LazyRandomTypeCollection::LazyRandomTypeCollection(
    const CVTypeArray &Types, uint32_t RecordCountHint,
    PartialOffsetArray PartialOffsets)
    : Database(RecordCountHint), Types(Types), DatabaseVisitor(Database),
      PartialOffsets(PartialOffsets) {
  KnownOffsets.resize(Database.capacity());
}

LazyRandomTypeCollection::LazyRandomTypeCollection(ArrayRef<uint8_t> Data,
                                                   uint32_t RecordCountHint)
    : LazyRandomTypeCollection(RecordCountHint) {
  reset(Data);
}

LazyRandomTypeCollection::LazyRandomTypeCollection(StringRef Data,
                                                   uint32_t RecordCountHint)
    : LazyRandomTypeCollection(
          makeArrayRef(Data.bytes_begin(), Data.bytes_end()), RecordCountHint) {
}

LazyRandomTypeCollection::LazyRandomTypeCollection(const CVTypeArray &Types,
                                                   uint32_t NumRecords)
    : LazyRandomTypeCollection(Types, NumRecords, PartialOffsetArray()) {}

void LazyRandomTypeCollection::reset(StringRef Data) {
  reset(makeArrayRef(Data.bytes_begin(), Data.bytes_end()));
}

void LazyRandomTypeCollection::reset(ArrayRef<uint8_t> Data) {
  PartialOffsets = PartialOffsetArray();

  BinaryStreamReader Reader(Data, support::little);
  error(Reader.readArray(Types, Reader.getLength()));

  KnownOffsets.resize(Database.capacity());
}

CVType LazyRandomTypeCollection::getType(TypeIndex Index) {
  error(ensureTypeExists(Index));
  return Database.getTypeRecord(Index);
}

StringRef LazyRandomTypeCollection::getTypeName(TypeIndex Index) {
  if (!Index.isSimple()) {
    // Try to make sure the type exists.  Even if it doesn't though, it may be
    // because we're dumping a symbol stream with no corresponding type stream
    // present, in which case we still want to be able to print <unknown UDT>
    // for the type names.
    consumeError(ensureTypeExists(Index));
  }

  return Database.getTypeName(Index);
}

bool LazyRandomTypeCollection::contains(TypeIndex Index) {
  return Database.contains(Index);
}

uint32_t LazyRandomTypeCollection::size() { return Database.size(); }

uint32_t LazyRandomTypeCollection::capacity() { return Database.capacity(); }

Error LazyRandomTypeCollection::ensureTypeExists(TypeIndex TI) {
  if (!Database.contains(TI)) {
    if (auto EC = visitRangeForType(TI))
      return EC;
  }
  return Error::success();
}

Error LazyRandomTypeCollection::visitRangeForType(TypeIndex TI) {
  if (PartialOffsets.empty())
    return fullScanForType(TI);

  auto Next = std::upper_bound(PartialOffsets.begin(), PartialOffsets.end(), TI,
                               [](TypeIndex Value, const TypeIndexOffset &IO) {
                                 return Value < IO.Type;
                               });

  assert(Next != PartialOffsets.begin());
  auto Prev = std::prev(Next);

  TypeIndex TIB = Prev->Type;
  if (Database.contains(TIB)) {
    // They've asked us to fetch a type index, but the entry we found in the
    // partial offsets array has already been visited.  Since we visit an entire
    // block every time, that means this record should have been previously
    // discovered.  Ultimately, this means this is a request for a non-existant
    // type index.
    return make_error<CodeViewError>("Invalid type index");
  }

  TypeIndex TIE;
  if (Next == PartialOffsets.end()) {
    TIE = TypeIndex::fromArrayIndex(Database.capacity());
  } else {
    TIE = Next->Type;
  }

  if (auto EC = visitRange(TIB, Prev->Offset, TIE))
    return EC;
  return Error::success();
}

Optional<TypeIndex> LazyRandomTypeCollection::getFirst() {
  TypeIndex TI = TypeIndex::fromArrayIndex(0);
  if (auto EC = ensureTypeExists(TI)) {
    consumeError(std::move(EC));
    return None;
  }
  return TI;
}

Optional<TypeIndex> LazyRandomTypeCollection::getNext(TypeIndex Prev) {
  // We can't be sure how long this type stream is, given that the initial count
  // given to the constructor is just a hint.  So just try to make sure the next
  // record exists, and if anything goes wrong, we must be at the end.
  if (auto EC = ensureTypeExists(Prev + 1)) {
    consumeError(std::move(EC));
    return None;
  }

  return Prev + 1;
}

Error LazyRandomTypeCollection::fullScanForType(TypeIndex TI) {
  assert(PartialOffsets.empty());

  TypeIndex CurrentTI = TypeIndex::fromArrayIndex(0);
  uint32_t Offset = 0;
  auto Begin = Types.begin();

  if (!Database.empty()) {
    // In the case of type streams which we don't know the number of records of,
    // it's possible to search for a type index triggering a full scan, but then
    // later additional records are added since we didn't know how many there
    // would be until we did a full visitation, then you try to access the new
    // type triggering another full scan.  To avoid this, we assume that if the
    // database has some records, this must be what's going on.  So we ask the
    // database for the largest type index less than the one we're searching for
    // and only do the forward scan from there.
    auto Prev = Database.largestTypeIndexLessThan(TI);
    assert(Prev.hasValue() && "Empty database with valid types?");
    Offset = KnownOffsets[Prev->toArrayIndex()];
    CurrentTI = *Prev;
    ++CurrentTI;
    Begin = Types.at(Offset);
    ++Begin;
    Offset = Begin.offset();
  }

  auto End = Types.end();
  while (Begin != End) {
    if (auto EC = visitOneRecord(CurrentTI, Offset, *Begin))
      return EC;

    Offset += Begin.getRecordLength();
    ++Begin;
    ++CurrentTI;
  }
  if (CurrentTI <= TI) {
    return make_error<CodeViewError>("Type Index does not exist!");
  }
  return Error::success();
}

Error LazyRandomTypeCollection::visitRange(TypeIndex Begin,
                                           uint32_t BeginOffset,
                                           TypeIndex End) {

  auto RI = Types.at(BeginOffset);
  assert(RI != Types.end());

  while (Begin != End) {
    if (auto EC = visitOneRecord(Begin, BeginOffset, *RI))
      return EC;

    BeginOffset += RI.getRecordLength();
    ++Begin;
    ++RI;
  }

  return Error::success();
}

Error LazyRandomTypeCollection::visitOneRecord(TypeIndex TI, uint32_t Offset,
                                               CVType &Record) {
  assert(!Database.contains(TI));
  if (auto EC = codeview::visitTypeRecord(Record, TI, DatabaseVisitor))
    return EC;
  // Keep the KnownOffsets array the same size as the Database's capacity. Since
  // we don't always know how many records are in the type stream, we need to be
  // prepared for the database growing and receicing a type index that can't fit
  // in our current buffer.
  if (KnownOffsets.size() < Database.capacity())
    KnownOffsets.resize(Database.capacity());
  KnownOffsets[TI.toArrayIndex()] = Offset;
  return Error::success();
}
