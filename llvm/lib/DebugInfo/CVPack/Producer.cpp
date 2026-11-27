//===- Producer.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Packs COFF objects. Every packed object references one base pack, found in
// the pack directories: the type records, global S_UDT/S_CONSTANT symbols and
// function debug info that the base holds move out of the objects, the rest
// stays in them, and no other pack is written. Without a base, or when the
// base no longer fits the inputs, the run writes a new base, <id>.cvp, from
// what occurs in more than one input; a rebase always does. The function
// debug info of inline functions and templates is the COMDAT .debug$S content
// that every object emitting the function repeats: a packed object keeps a
// 24-byte reference and the relocations. Packed objects are written to the
// staging directory.
//
// Records are identified by a 128-bit key over the record with every type
// index replaced by the referenced record's global hash. Function debug info
// is identified by its bytes with every type index replaced by the record's
// key id and every file checksum offset by the file's key id. A record is only
// packed when everything it references is packed too, so the records of a pack
// form a closed set and every pack record can be renumbered into the pack's own
// index space. Packed objects keep the original index numbering: the range map
// says which local indices come from the pack.
//
// The phases keep little per-record state, so that a whole game build fits in
// memory: a key id and a pack index per type record. Objects are reopened by
// each phase that needs their bytes.
//
//   1. Read, hash and key every object (parallel; keys interned on the fly).
//   2. Load the packs of the pack directories (the dedup set) and choose the
//      base: the pack that holds the most bytes of what the inputs contain.
//   3. Select the records to pack (parallel), measuring how well the base
//      fits; when it does not, start over as a rebase. A rebase then assigns
//      pack indices in object order (serial). First-seen order is kept:
//      reordering the pack by co-occurrence was measured to remove under 4%
//      of range-map entries, because objects use shared records in different
//      orders.
//   4. On a rebase, write the new base.
//   5. Stage every object (parallel), verifying that it materializes back to
//      its input byte for byte. An object that fails is staged unchanged.
//
// The base no longer fits when a rebase would pack more than a set fraction of
// the base that the base lacks, or the inputs use too little of the base.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CVPack/Producer.h"
#include "COFFSectionTable.h"
#include "CVPackUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/DebugInfo/CVPack/COFFEdit.h"
#include "llvm/DebugInfo/CVPack/FunctionDebug.h"
#include "llvm/DebugInfo/CVPack/Materialize.h"
#include "llvm/DebugInfo/CVPack/PackFile.h"
#include "llvm/DebugInfo/CVPack/PackIndex.h"
#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/SymbolRecordHelpers.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <set>

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::cvpack;
using namespace llvm::support::endian;

constexpr uint32_t NoIndex = std::numeric_limits<uint32_t>::max();
/// Marks a symbol selected for packing before it has an ordinal.
constexpr uint32_t Selected = NoIndex - 1;
/// Before assignment, InputObject::Mapped holds each record's key id, with
/// SelectedBit set once the record is selected; NoKey means not keyable.
constexpr uint32_t SelectedBit = 0x80000000;
constexpr uint32_t NoKey = 0x7fffffff;
constexpr uint16_t BLAKE3HashAlg = 2; // GlobalTypeHashAlg::BLAKE3

namespace {
struct KeyInfo {
  static RecordKey getEmptyKey() {
    RecordKey K;
    K.fill(0xff);
    return K;
  }
  static RecordKey getTombstoneKey() {
    RecordKey K;
    K.fill(0xfe);
    return K;
  }
  static unsigned getHashValue(const RecordKey &K) {
    return support::endian::read32le(K.data());
  }
  static bool isEqual(const RecordKey &A, const RecordKey &B) { return A == B; }
};

/// A message, written like a stream and handed to a callback of PackOptions
/// when it is complete. Nothing is reported without a callback.
class Message {
public:
  explicit Message(const std::function<void(const Twine &)> &Sink)
      : Sink(Sink) {}
  ~Message() {
    if (!Sink)
      return;
    // The callbacks of PackOptions are called from one thread at a time.
    static std::mutex Mutex;
    std::lock_guard<std::mutex> Lock(Mutex);
    Sink(Text);
  }
  template <typename T> Message &operator<<(const T &Value) {
    OS << Value;
    return *this;
  }

private:
  const std::function<void(const Twine &)> &Sink;
  std::string Text;
  raw_string_ostream OS{Text};
};

/// Interns 128-bit record keys into dense ids from many threads. The ids
/// depend on thread timing; nothing written depends on them.
class KeyTable {
public:
  uint32_t intern(const RecordKey &Key) {
    Shard &S = Shards[Key[15]];
    std::lock_guard<std::mutex> Lock(S.Mutex);
    auto [It, Inserted] = S.Ids.try_emplace(Key, 0);
    if (Inserted)
      It->second = Next.fetch_add(1, std::memory_order_relaxed);
    return It->second;
  }
  uint32_t size() const { return Next.load(); }
  /// Visit every key and its id. Not thread-safe with intern().
  template <typename Fn> void forEach(Fn Visit) const {
    for (const Shard &S : Shards)
      for (const auto &[Key, Id] : S.Ids)
        Visit(Key, Id);
  }

private:
  struct Shard {
    std::mutex Mutex;
    DenseMap<RecordKey, uint32_t, KeyInfo> Ids;
  };
  std::array<Shard, 256> Shards;
  std::atomic<uint32_t> Next{0};
};

/// Limits the bytes of input objects being processed at once, which bounds
/// the transient buffers of the parallel phases. An object larger than the
/// limit runs alone.
class MemoryBudget {
public:
  explicit MemoryBudget(uint64_t Limit) : Limit(Limit) {}
  void acquire(uint64_t N) {
    std::unique_lock<std::mutex> Lock(Mutex);
    Cond.wait(Lock, [&] { return InFlight == 0 || InFlight + N <= Limit; });
    InFlight += N;
  }
  void release(uint64_t N) {
    {
      std::lock_guard<std::mutex> Lock(Mutex);
      InFlight -= N;
    }
    Cond.notify_all();
  }

private:
  std::mutex Mutex;
  std::condition_variable Cond;
  uint64_t InFlight = 0;
  const uint64_t Limit;
};

/// Counts objects per key, saturating at 2: only "shared or not" matters.
class SharedCounter {
public:
  void resize(size_t N) { Counts = std::vector<std::atomic<uint8_t>>(N); }
  void add(uint32_t Key) {
    if (Counts[Key].load(std::memory_order_relaxed) < 2)
      Counts[Key].fetch_add(1, std::memory_order_relaxed);
  }
  bool shared(uint32_t Key) const {
    return Counts[Key].load(std::memory_order_relaxed) >= 2;
  }
  bool present(uint32_t Key) const {
    return Counts[Key].load(std::memory_order_relaxed) >= 1;
  }

private:
  std::vector<std::atomic<uint8_t>> Counts;
};

struct SymbolRecordRef {
  uint32_t Offset = 0;          // record offset in the section
  uint32_t Size = 0;            // record size, including its prefix
  uint32_t Candidate = NoIndex; // index into InputObject::Symbols
};

/// One symbols subsection of a non-COMDAT .debug$S section.
struct SymbolSubsection {
  uint32_t Section = 0;      // one-based section index
  uint32_t HeaderOffset = 0; // offset of the subsection header in the section
  uint32_t DataSize = 0;
  std::vector<SymbolRecordRef> Records;
};

/// A global S_UDT or S_CONSTANT that may move to a pack.
struct SymbolCandidate {
  uint32_t Subsection = 0;
  uint32_t Record = 0;
  uint32_t KeyId = NoIndex;
  SmallVector<uint32_t, 2> Refs; // array indices of referenced local types
};

/// A COMDAT .debug$S section: the debug info of one function.
struct FunctionDebug {
  uint32_t Section = 0;
  uint32_t Size = 0; // raw size of the section
  uint32_t KeyId = NoIndex;
  uint32_t FirstType = 0, NumTypes = 0; // in InputObject::FunctionTypes
  uint32_t FirstFile = 0, NumFiles = 0; // in InputObject::FunctionFiles
};

struct InputObject {
  std::string Path;
  std::string SkipReason; // nonempty: stage this object unchanged
  uint64_t Size = 0;      // size of the original object
  uint32_t NumTypes = 0;  // type records in the original object
  uint16_t Machine = 0;

  bool IsPackedInput = false;
  std::optional<PackId> InputPack; // LF_CVPACK_LIST of a packed input
  // A packed input whose pack is the base is staged unchanged.
  bool PassThrough = false;
  // A packed input that could not be restored is staged as it is, still
  // packed.
  bool Unrestored = false;
  // Original bytes of a packed input that is packed again.
  std::vector<uint8_t> Materialized;

  uint32_t TypeSection = 0;
  uint32_t HashSection = 0;
  bool DropDebugH = false;
  // Records whose .debug$H hash differs from the recomputed one.
  std::vector<uint32_t> DebugHPatches;
  std::vector<SymbolSubsection> Subsections;
  std::vector<SymbolCandidate> Symbols;

  // One entry per type record: the key id (NoKey: not keyable), with
  // SelectedBit once selected; after assignment, the pack type index, or 0
  // for an inline record.
  std::vector<uint32_t> Mapped;
  // Selected, then the pack symbol ordinal; NoIndex = inline.
  std::vector<uint32_t> SymbolMapped;

  // COMDAT .debug$S sections that may move to the pack, with the distinct
  // local type records (array indices) and file key ids each one references.
  std::vector<FunctionDebug> Functions;
  std::vector<uint32_t> FunctionTypes;
  std::vector<uint32_t> FunctionFiles;
  // Per function: Selected, then the pack function ordinal; NoIndex = inline.
  std::vector<uint32_t> FunctionMapped;
  // COMDAT .debug$S sections that cannot move, by reason: count and bytes.
  std::map<std::string, std::pair<uint64_t, uint64_t>> FunctionSkips;

  bool packable() const { return SkipReason.empty() && !PassThrough; }
};

/// The bytes of an object as needed by one phase.
struct ObjectView {
  std::unique_ptr<MemoryBuffer> Buffer;
  std::vector<uint8_t> Owned;
  ArrayRef<uint8_t> Bytes;
  COFFSectionTable Sections;
  std::vector<ArrayRef<uint8_t>> Types;

  /// The raw data of a section, empty if it is invalid.
  ArrayRef<uint8_t> section(uint32_t Index) const {
    return Sections.contents(Index).value_or(ArrayRef<uint8_t>());
  }

  /// The contents of the COMDAT or the other .debug$S sections.
  std::vector<ArrayRef<uint8_t>> debugSSections(bool Comdat) const {
    std::vector<ArrayRef<uint8_t>> Result;
    for (uint32_t I = 1; I <= Sections.size(); ++I)
      if (Sections.name(I) == ".debug$S" && Sections.isComdat(I) == Comdat)
        Result.push_back(section(I));
    return Result;
  }
};
} // namespace

static RecordKey hashKey(ArrayRef<uint8_t> Data) {
  return BLAKE3::hash<16>(Data);
}

static uint32_t internFile(KeyTable &FileKeys, const PackFileEntry &File) {
  std::string Identity = fileIdentity(File);
  return FileKeys.intern(hashKey(arrayRefFromStringRef(Identity)));
}

static RecordKey fileKey(const PackFileEntry &File) {
  std::string Identity = fileIdentity(File);
  return hashKey(arrayRefFromStringRef(Identity));
}

/// The key of a function's debug info: its content with every type index and
/// file reference replaced by a marker, followed by the keys of the referenced
/// records and files in field order. It depends on no object's numbering, so a
/// pack index can store it.
static Expected<RecordKey>
functionKey(ArrayRef<uint8_t> Content, std::vector<uint8_t> &Scratch,
            function_ref<Expected<RecordKey>(uint32_t)> TypeKey,
            function_ref<Expected<RecordKey>(uint32_t)> FileKey) {
  Scratch.clear();
  std::vector<uint8_t> Keys;
  if (Error E = transformFunctionDebug(
          Content, Scratch,
          [&](uint32_t Index, uint32_t) -> Expected<uint32_t> {
            auto Key = TypeKey(Index);
            if (!Key)
              return Key.takeError();
            Keys.insert(Keys.end(), Key->begin(), Key->end());
            // Differs from every simple type index.
            return 0x80000000u;
          },
          [&](uint32_t File, uint32_t) -> Expected<uint32_t> {
            auto Key = FileKey(File);
            if (!Key)
              return Key.takeError();
            Keys.insert(Keys.end(), Key->begin(), Key->end());
            return 0;
          },
          FileOperandEncoding::Fixed))
    return std::move(E);
  Scratch.insert(Scratch.end(), Keys.begin(), Keys.end());
  return hashKey(Scratch);
}

static Error producerError(const Twine &Message) {
  return createStringError(inconvertibleErrorCode(), Message);
}

static std::string idString(const PackId &Id) {
  return toHex(Id, /*LowerCase=*/true);
}

static std::optional<PackId> parseIdString(StringRef Hex) {
  std::string Bytes;
  PackId Id;
  if (Hex.size() != 2 * Id.size() || !tryGetFromHex(Hex, Bytes))
    return std::nullopt;
  llvm::copy(Bytes, Id.begin());
  return Id;
}

/// Hash type records in one pass. A record whose references cannot be
/// resolved (a forward reference) gets an empty hash, and so does everything
/// that references it.
static std::vector<GloballyHashedType>
hashTypesOnePass(ArrayRef<CVType> Types) {
  std::vector<GloballyHashedType> Hashes;
  Hashes.reserve(Types.size());
  for (CVType Type : Types)
    Hashes.push_back(GloballyHashedType::hashType(Type, Hashes, Hashes));
  return Hashes;
}

/// Collect the non-simple type references of a record as array indices.
/// Returns false if a reference is not strictly backwards.
template <typename VectorT>
static bool collectRefs(ArrayRef<uint8_t> Record, bool IsSymbol, uint32_t Self,
                        VectorT &Out) {
  bool Backwards = true;
  Error E = forEachTypeIndex(Record, IsSymbol, [&](uint32_t Offset, TiRefKind) {
    uint32_t Index = read32le(Record.data() + Offset);
    if (Index < TypeIndex::FirstNonSimpleIndex)
      return Error::success();
    uint32_t ArrayIndex = Index - TypeIndex::FirstNonSimpleIndex;
    if (ArrayIndex >= Self)
      Backwards = false;
    else
      Out.push_back(ArrayIndex);
    return Error::success();
  });
  if (E) {
    consumeError(std::move(E));
    return false;
  }
  return Backwards;
}

/// Rewrite the non-simple type indices of a record through Map, which is
/// indexed by array index.
static std::vector<uint8_t> remapRecord(ArrayRef<uint8_t> Record, bool IsSymbol,
                                        ArrayRef<uint32_t> Map) {
  std::vector<uint8_t> Result(Record.begin(), Record.end());
  // The record was keyed, so its type indices are known.
  cantFail(forEachTypeIndex(Result, IsSymbol, [&](uint32_t Offset, TiRefKind) {
    uint32_t Index = read32le(Result.data() + Offset);
    if (Index >= TypeIndex::FirstNonSimpleIndex)
      support::endian::write32le(Result.data() + Offset,
                                 Map[Index - TypeIndex::FirstNonSimpleIndex]);
    return Error::success();
  }));
  return Result;
}

/// Split a C13 type stream into records.
static Error splitTypes(ArrayRef<uint8_t> Section,
                        std::vector<ArrayRef<uint8_t>> &Out) {
  if (Section.size() < 4 ||
      read32le(Section.data()) != COFF::DEBUG_SECTION_MAGIC)
    return producerError("no CodeView C13 type stream");
  return forEachCodeViewRecord<CVType>(
      Section.drop_front(sizeof(uint32_t)), [&](CVType Type) -> Error {
        if (Type.length() < sizeof(RecordPrefix))
          return producerError("truncated type record");
        Out.push_back(Type.RecordData);
        return Error::success();
      });
}

/// Open the original bytes of an analyzed object and split its type stream.
static Error openView(const InputObject &Object, ObjectView &View,
                      CVPackResolver *Resolver = nullptr) {
  if (!Object.Materialized.empty()) {
    View.Bytes = Object.Materialized;
  } else if (Object.IsPackedInput) {
    // A packed input staged unchanged keeps no copy of its original bytes.
    if (!Resolver)
      return producerError("packed input reopened without its packs");
    auto Buffer = MemoryBuffer::getFile(Object.Path, /*IsText=*/false,
                                        /*RequiresNullTerminator=*/false);
    if (!Buffer)
      return errorCodeToError(Buffer.getError());
    auto Original =
        materializeObject(arrayRefFromStringRef((*Buffer)->getBuffer()),
                          *Resolver, sys::path::parent_path(Object.Path));
    if (!Original)
      return Original.takeError();
    View.Owned = std::move(*Original);
    View.Bytes = View.Owned;
  } else {
    auto Buffer = MemoryBuffer::getFile(Object.Path, /*IsText=*/false,
                                        /*RequiresNullTerminator=*/false);
    if (!Buffer)
      return errorCodeToError(Buffer.getError());
    View.Buffer = std::move(*Buffer);
    View.Bytes = arrayRefFromStringRef(View.Buffer->getBuffer());
  }
  if (View.Bytes.size() != Object.Size)
    return producerError("input changed while it was being packed");
  auto Sections = COFFSectionTable::create(View.Bytes);
  if (!Sections)
    return Sections.takeError();
  View.Sections = std::move(*Sections);
  View.Types.clear();
  View.Types.reserve(Object.NumTypes);
  if (Error E = splitTypes(View.section(Object.TypeSection), View.Types))
    return E;
  if (View.Types.size() != Object.NumTypes)
    return producerError("input changed while it was being packed");
  return Error::success();
}

/// Phase 1: read one input object, key its records and find its symbol
/// candidates.
static void analyzeObject(InputObject &Object, CVPackResolver &Resolver,
                          KeyTable &TypeKeys, KeyTable &SymbolKeys,
                          KeyTable &FunctionKeys, KeyTable &FileKeys) {
  auto fail = [&](const Twine &Reason) { Object.SkipReason = Reason.str(); };
  auto Buffer = MemoryBuffer::getFile(Object.Path, /*IsText=*/false,
                                      /*RequiresNullTerminator=*/false);
  if (!Buffer) {
    fail("cannot read: " + Buffer.getError().message());
    return;
  }
  ArrayRef<uint8_t> Bytes = arrayRefFromStringRef((*Buffer)->getBuffer());

  auto Sections = COFFSectionTable::create(Bytes);
  if (!Sections) {
    fail(toString(Sections.takeError()));
    return;
  }

  // An input that is already packed is materialized first, so that packing is
  // always computed from the original records. Its pack is found next to it,
  // or in the pack directories.
  for (uint32_t I = 1; I <= Sections->size(); ++I) {
    if (Sections->name(I) != ".debug$T")
      continue;
    auto Data = Sections->contents(I);
    if (Data && isPackedTypeStream(*Data)) {
      auto Original = materializeObject(Bytes, Resolver,
                                        sys::path::parent_path(Object.Path));
      if (!Original) {
        fail("cannot materialize packed input: " +
             toString(Original.takeError()));
        Object.Unrestored = true;
        return;
      }
      size_t ListSize = size_t(read16le(Data->data() + 4)) + 2;
      if (ListSize <= Data->size() - 4) {
        if (auto Id = decodePackList(CVType(Data->slice(4, ListSize))))
          Object.InputPack = *Id;
        else
          consumeError(Id.takeError());
      }
      Object.IsPackedInput = true;
      Object.Materialized = std::move(*Original);
      Bytes = Object.Materialized;
      Sections = COFFSectionTable::create(Bytes);
      if (!Sections) {
        fail(toString(Sections.takeError()));
        return;
      }
    }
    break;
  }
  Object.Size = Bytes.size();
  Object.Machine = Sections->layout().Machine;

  std::vector<uint32_t> DebugSSections;
  for (uint32_t I = 1; I <= Sections->size(); ++I) {
    StringRef Name = Sections->name(I);
    if (Name == ".debug$T") {
      if (Object.TypeSection != 0) {
        fail("multiple .debug$T sections");
        return;
      }
      Object.TypeSection = I;
    } else if (Name == ".debug$H") {
      Object.HashSection = Object.HashSection == 0 ? I : NoIndex;
    } else if (Name == ".debug$P") {
      // A precompiled header object (/Yc): other objects refer to its types.
      fail("precompiled header object");
      return;
    } else if (Name == ".debug$S" && !Sections->isComdat(I)) {
      DebugSSections.push_back(I);
    }
  }
  if (Object.TypeSection == 0) {
    fail("no .debug$T section");
    return;
  }
  if (Sections->header(Object.TypeSection).NumberOfRelocations != 0) {
    fail(".debug$T has relocations");
    return;
  }
  auto TypeData = Sections->contents(Object.TypeSection);
  std::vector<ArrayRef<uint8_t>> Records;
  if (!TypeData) {
    fail("no CodeView C13 type stream");
    return;
  }
  if (Error E = splitTypes(*TypeData, Records)) {
    fail(toString(std::move(E)));
    return;
  }
  // The type indices of an object that refers to a type server (/Zi) or to a
  // precompiled header object (/Yu) do not only number its own records.
  if (!Records.empty()) {
    TypeLeafKind FirstKind = CVType(Records.front()).kind();
    if (FirstKind == LF_TYPESERVER2 || FirstKind == LF_PRECOMP) {
      fail(FirstKind == LF_TYPESERVER2
               ? "refers to a type server"
               : "refers to a precompiled header object");
      return;
    }
  }
  std::vector<CVType> Types(Records.begin(), Records.end());
  std::vector<GloballyHashedType> Hashes = hashTypesOnePass(Types);

  // .debug$H is dropped when it can be recomputed. Hashes that differ from
  // the recomputed ones, as an older compiler wrote for some LF_VTSHAPE
  // records, are kept in the origin record, up to a limit.
  if (Object.HashSection != 0 && Object.HashSection != NoIndex &&
      Sections->header(Object.HashSection).NumberOfRelocations == 0) {
    auto HashData = Sections->contents(Object.HashSection);
    if (HashData && HashData->size() == 8 + 8 * Hashes.size() &&
        read32le(HashData->data()) == COFF::DEBUG_HASHES_SECTION_MAGIC &&
        read16le(HashData->data() + 4) == 0 &&
        read16le(HashData->data() + 6) == BLAKE3HashAlg &&
        llvm::none_of(Hashes,
                      [](const GloballyHashedType &H) { return H.empty(); })) {
      constexpr size_t MaxPatches = 2000;
      for (size_t I = 0;
           I != Hashes.size() && Object.DebugHPatches.size() <= MaxPatches; ++I)
        if (!std::equal(Hashes[I].Hash.begin(), Hashes[I].Hash.end(),
                        HashData->begin() + 8 + I * 8))
          Object.DebugHPatches.push_back(I);
      Object.DropDebugH = Object.DebugHPatches.size() <= MaxPatches;
      if (!Object.DropDebugH)
        Object.DebugHPatches = {};
    }
  }

  size_t Count = Types.size();
  Object.NumTypes = Count;
  Object.Mapped.assign(Count, NoKey);
  std::vector<RecordKey> KeyValues(Count);
  SmallVector<uint32_t, 16> Refs;
  for (uint32_t I = 0; I != Count; ++I) {
    TypeLeafKind Kind = Types[I].kind();
    if (Kind == TypeLeafKind::LF_CVPACK_LIST ||
        Kind == TypeLeafKind::LF_CVPACK_ORIGIN ||
        Kind == TypeLeafKind::LF_CVPACK_MAP || Hashes[I].empty())
      continue;
    Refs.clear();
    if (!collectRefs(Records[I], false, I, Refs))
      continue;
    auto Key = getTypeRecordKey(Records[I], Hashes, Hashes);
    if (!Key) {
      consumeError(Key.takeError());
      continue;
    }
    Object.Mapped[I] = TypeKeys.intern(*Key);
    KeyValues[I] = *Key;
  }

  // Sections that COFF symbols point into, other than at offset 0, cannot have
  // their layout changed.
  DenseSet<uint32_t> PinnedSections;
  {
    // COFFObjectFile gives no symbols for a symbol table it cannot read.
    const object::COFFObjectFile &Obj = Sections->object();
    if (Obj.getPointerToSymbolTable() != 0 &&
        Obj.getNumberOfSymbols() == Obj.getRawNumberOfSymbols()) {
      for (uint32_t I = 0, E = Obj.getNumberOfSymbols(); I < E; ++I) {
        object::COFFSymbolRef Symbol = cantFail(Obj.getSymbol(I));
        if (Symbol.getSectionNumber() > 0 && Symbol.getValue() != 0)
          PinnedSections.insert(Symbol.getSectionNumber());
        I += Symbol.getNumberOfAuxSymbols();
      }
    } else {
      PinnedSections.insert(DebugSSections.begin(), DebugSSections.end());
    }
  }

  for (uint32_t Section : DebugSSections) {
    if (PinnedSections.contains(Section))
      continue;
    auto Data = Sections->contents(Section);
    if (!Data || Data->size() < 4 ||
        read32le(Data->data()) != COFF::DEBUG_SECTION_MAGIC)
      continue;

    // Relocation targets inside the section; packed symbols must avoid them.
    auto Relocs = Sections->relocations(Section);
    if (!Relocs)
      continue;
    std::vector<uint32_t> Relocations;
    for (const object::coff_relocation &R : *Relocs)
      Relocations.push_back(R.VirtualAddress);
    llvm::sort(Relocations);
    auto touchesRelocation = [&](uint32_t Begin, uint32_t End) {
      auto It = llvm::lower_bound(Relocations, Begin > 8 ? Begin - 8 : 0);
      for (; It != Relocations.end() && *It < End; ++It)
        if (*It + 2 > Begin)
          return true;
      return false;
    };

    size_t Offset = 4;
    while (Offset + 8 <= Data->size()) {
      uint32_t Kind = read32le(Data->data() + Offset);
      uint32_t Size = read32le(Data->data() + Offset + 4);
      if (Size > Data->size() - Offset - 8)
        break;
      if (Kind == uint32_t(DebugSubsectionKind::Symbols)) {
        SymbolSubsection Subsection;
        Subsection.Section = Section;
        Subsection.HeaderOffset = Offset;
        Subsection.DataSize = Size;
        size_t FirstCandidate = Object.Symbols.size();
        int Depth = 0;
        bool Valid = true;
        size_t Pos = Offset + 8;
        size_t End = Offset + 8 + Size;
        while (Pos < End) {
          if (End - Pos < 4) {
            Valid = false;
            break;
          }
          uint32_t RecordSize = read16le(Data->data() + Pos) + 2;
          if (RecordSize < 4 || RecordSize > End - Pos) {
            Valid = false;
            break;
          }
          SymbolKind SymKind =
              static_cast<SymbolKind>(read16le(Data->data() + Pos + 2));
          if (SymKind == SymbolKind::S_CVPACK_MAP) {
            Valid = false;
            break;
          }
          SymbolRecordRef Record{static_cast<uint32_t>(Pos), RecordSize,
                                 NoIndex};
          if (Depth == 0 &&
              (SymKind == SymbolKind::S_UDT ||
               SymKind == SymbolKind::S_CONSTANT) &&
              RecordSize % 4 == 0 &&
              !touchesRelocation(Pos, Pos + RecordSize)) {
            SymbolCandidate Candidate;
            ArrayRef<uint8_t> SymBytes = Data->slice(Pos, RecordSize);
            auto Key = getSymbolRecordKey(SymBytes, Hashes, Hashes);
            if (Key && collectRefs(SymBytes, true, Count, Candidate.Refs)) {
              Candidate.Subsection = Object.Subsections.size();
              Candidate.Record = Subsection.Records.size();
              Candidate.KeyId = SymbolKeys.intern(*Key);
              Record.Candidate = Object.Symbols.size();
              Object.Symbols.push_back(std::move(Candidate));
            } else if (!Key) {
              consumeError(Key.takeError());
            }
          }
          if (symbolOpensScope(SymKind))
            ++Depth;
          else if (symbolEndsScope(SymKind))
            --Depth;
          Subsection.Records.push_back(Record);
          Pos += RecordSize;
        }
        if (!Valid)
          Object.Symbols.resize(FirstCandidate);
        else
          Object.Subsections.push_back(std::move(Subsection));
      }
      Offset += 8 + ((Size + 3) & ~size_t(3));
    }
  }

  // COMDAT .debug$S sections: the debug info of inline functions and
  // templates, repeated in every object that emits the function.
  std::vector<uint32_t> ComdatDebugS;
  for (uint32_t I = 1; I <= Sections->size(); ++I)
    if (Sections->name(I) == ".debug$S" && Sections->isComdat(I))
      ComdatDebugS.push_back(I);
  if (ComdatDebugS.empty())
    return;
  std::vector<ArrayRef<uint8_t>> MainSections;
  for (uint32_t Section : DebugSSections)
    if (auto Data = Sections->contents(Section))
      MainSections.push_back(*Data);
  auto skip = [&](const Twine &Reason, uint64_t Bytes) {
    auto &Entry = Object.FunctionSkips[Reason.str()];
    ++Entry.first;
    Entry.second += Bytes;
  };
  auto Files = ObjectFileTable::create(MainSections);
  if (!Files) {
    for (uint32_t Section : ComdatDebugS)
      skip(toString(Files.takeError()),
           Sections->header(Section).SizeOfRawData);
    return;
  }
  DenseMap<uint32_t, uint32_t> FileKeyAt; // checksum offset -> file key id
  std::vector<uint8_t> Normalized, Restored;
  std::vector<uint32_t> Relocations;
  SmallVector<uint32_t, 32> TypeRefs, FileRefs;
  for (uint32_t Section : ComdatDebugS) {
    auto Data = Sections->contents(Section);
    if (!Data || Data->size() <= FunctionRefSectionSize ||
        read32le(Data->data()) != COFF::DEBUG_SECTION_MAGIC) {
      skip("empty or not CodeView", Data ? Data->size() : 0);
      continue;
    }
    if (PinnedSections.contains(Section) ||
        (Sections->header(Section).Characteristics &
         COFF::IMAGE_SCN_LNK_NRELOC_OVFL)) {
      skip("pinned or relocation overflow", Data->size());
      continue;
    }
    auto Relocs = Sections->relocations(Section);
    if (!Relocs) {
      skip("invalid relocations", Data->size());
      continue;
    }
    Relocations.clear();
    for (const object::coff_relocation &R : *Relocs)
      Relocations.push_back(R.VirtualAddress);
    llvm::sort(Relocations);
    // A relocation over a rewritten field would make the pack copy differ.
    auto touchesRelocation = [&](uint32_t Field) {
      auto It = llvm::lower_bound(Relocations, Field > 3 ? Field - 3 : 0);
      return It != Relocations.end() && *It < Field + 4;
    };

    ArrayRef<uint8_t> Content = Data->drop_front(sizeof(uint32_t));
    // Type indices become key ids, tagged with the high bit so that they
    // differ from simple type indices; file checksum offsets become file key
    // ids. Offsets count from the end of the magic.
    TypeRefs.clear();
    FileRefs.clear();
    Normalized.clear();
    Error E = transformFunctionDebug(
        Content, Normalized,
        [&](uint32_t Index, uint32_t Offset) -> Expected<uint32_t> {
          if (touchesRelocation(Offset + sizeof(uint32_t)))
            return producerError("relocation over a rewritten field");
          uint32_t ArrayIndex = Index - TypeIndex::FirstNonSimpleIndex;
          if (ArrayIndex >= Count || Object.Mapped[ArrayIndex] == NoKey)
            return producerError("unkeyed type reference");
          TypeRefs.push_back(ArrayIndex);
          return Object.Mapped[ArrayIndex] | SelectedBit;
        },
        [&](uint32_t Checksum, uint32_t Offset) -> Expected<uint32_t> {
          if (touchesRelocation(Offset + sizeof(uint32_t)))
            return producerError("relocation over a rewritten field");
          auto [It, Inserted] = FileKeyAt.try_emplace(Checksum, NoIndex);
          if (Inserted) {
            std::optional<PackFileEntry> Entry = Files->entryAt(Checksum);
            if (Entry && Files->lookup(*Entry) == Checksum)
              It->second = internFile(FileKeys, *Entry);
          }
          if (It->second == NoIndex)
            return producerError("unknown file reference");
          FileRefs.push_back(It->second);
          return It->second;
        },
        FileOperandEncoding::Fixed);
    // Inline sites that change files are re-encoded when restored; they must
    // come back as they are.
    if (!E) {
      Restored.clear();
      E = transformFunctionDebug(
          Content, Restored,
          [](uint32_t I, uint32_t) -> Expected<uint32_t> { return I; },
          [](uint32_t F, uint32_t) -> Expected<uint32_t> { return F; },
          FileOperandEncoding::Minimal);
      if (!E && ArrayRef<uint8_t>(Restored) != Content)
        E = producerError("inline-site annotations are not minimally encoded");
    }
    if (E) {
      skip(toString(std::move(E)), Data->size());
      continue;
    }
    llvm::sort(TypeRefs);
    TypeRefs.erase(llvm::unique(TypeRefs), TypeRefs.end());
    llvm::sort(FileRefs);
    FileRefs.erase(llvm::unique(FileRefs), FileRefs.end());
    auto Key = functionKey(
        Content, Normalized,
        [&](uint32_t Index) -> Expected<RecordKey> {
          return KeyValues[Index - TypeIndex::FirstNonSimpleIndex];
        },
        [&](uint32_t Offset) -> Expected<RecordKey> {
          return fileKey(*Files->entryAt(Offset));
        });
    if (!Key) {
      skip(toString(Key.takeError()), Data->size());
      continue;
    }
    FunctionDebug Function;
    Function.Section = Section;
    Function.Size = Data->size();
    Function.KeyId = FunctionKeys.intern(*Key);
    Function.FirstType = Object.FunctionTypes.size();
    Function.NumTypes = TypeRefs.size();
    Object.FunctionTypes.insert(Object.FunctionTypes.end(), TypeRefs.begin(),
                                TypeRefs.end());
    Function.FirstFile = Object.FunctionFiles.size();
    Function.NumFiles = FileRefs.size();
    Object.FunctionFiles.insert(Object.FunctionFiles.end(), FileRefs.begin(),
                                FileRefs.end());
    Object.Functions.push_back(Function);
  }
}

namespace {
/// A CodeView pack in the dedup set, with its logical records keyed.
struct ExistingPack {
  /// The pack as loaded by the run's resolver, which also verifies the staged
  /// objects with it: each pack is read once.
  const ResolvedCVPack *Pack = nullptr;
  PackId Id{};
  std::vector<uint32_t> KeyIds;       // per logical record, NoIndex if unkeyed
  std::vector<uint32_t> SymbolKeyIds; // per own symbol
  std::vector<uint32_t> FunctionKeyIds; // per own function, NoIndex if unkeyed
};
} // namespace

/// Key a loaded pack: the records an object can refer to, with their keys.
static Expected<PackIndex> buildPackIndex(const ResolvedCVPack &Pack,
                                          const PackId &Id) {
  PackIndex Index;
  Index.PackContentId = Id;
  Index.PackFileSize = Pack.Buffer->getBufferSize();
  ArrayRef<CVType> Types = Pack.View.Types;
  size_t Count = Types.size();
  Index.TypeCount = Count;
  Index.SymbolCount = Pack.View.Symbols.size();
  Index.FunctionCount = Pack.View.Functions.size();

  std::vector<GloballyHashedType> Hashes;
  ArrayRef<uint8_t> Stored = Pack.View.GlobalHashes;
  if (!Stored.empty() && Stored.size() == 8 * Count) {
    Hashes.resize(Count);
    for (size_t I = 0; I != Count; ++I)
      std::copy_n(Stored.data() + 8 * I, 8, Hashes[I].Hash.begin());
  } else {
    Hashes = hashTypesOnePass(Types);
  }

  // A record can be referred to when it is the first of its key and every
  // record it references can be too.
  std::vector<RecordKey> Keys(Count);
  std::vector<uint8_t> Keyed(Count, 0), Importable(Count, 0);
  DenseSet<RecordKey, KeyInfo> Seen;
  SmallVector<uint32_t, 16> Refs;
  for (uint32_t I = 0; I != Count; ++I) {
    if (Hashes[I].empty())
      continue;
    auto Key = getTypeRecordKey(Types[I].RecordData, Hashes, Hashes);
    if (!Key) {
      consumeError(Key.takeError());
      continue;
    }
    Keys[I] = *Key;
    Keyed[I] = 1;
    if (!Seen.insert(*Key).second)
      continue;
    Refs.clear();
    if (!collectRefs(Types[I].RecordData, false, I, Refs) ||
        !llvm::all_of(Refs, [&](uint32_t R) { return Importable[R] != 0; }))
      continue;
    Importable[I] = 1;
    Index.Types.push_back({*Key, I});
  }

  DenseSet<RecordKey, KeyInfo> SeenSymbols;
  for (uint32_t S = 0; S != Pack.View.Symbols.size(); ++S) {
    auto Key = getSymbolRecordKey(Pack.View.Symbols[S].data(), Hashes, Hashes);
    if (!Key) {
      consumeError(Key.takeError());
      continue;
    }
    if (SeenSymbols.insert(*Key).second)
      Index.Symbols.push_back({*Key, S});
  }

  DenseSet<RecordKey, KeyInfo> SeenFunctions;
  std::vector<uint8_t> Scratch;
  for (uint32_t F = 0; F != Pack.View.Functions.size(); ++F) {
    auto Key = functionKey(
        Pack.View.Functions[F], Scratch,
        [&](uint32_t TypeIndexValue) -> Expected<RecordKey> {
          uint32_t I = TypeIndexValue - TypeIndex::FirstNonSimpleIndex;
          if (I >= Count || !Keyed[I])
            return producerError("unkeyed type reference");
          return Keys[I];
        },
        [&](uint32_t File) -> Expected<RecordKey> {
          if (File >= Pack.View.Files.size())
            return producerError("invalid pack file reference");
          return fileKey(Pack.View.Files[File]);
        });
    if (!Key) {
      consumeError(Key.takeError());
      continue;
    }
    if (SeenFunctions.insert(*Key).second)
      Index.Functions.push_back({*Key, F});
  }
  return Index;
}

/// Give the pack's records the key ids of the inputs that hold the same keys.
/// Records no input holds keep NoIndex: they cannot be referred to anyway.
static void applyPackIndex(const PackIndexView &Index, ExistingPack &E,
                           const KeyTable &TypeKeys, const KeyTable &SymbolKeys,
                           const KeyTable &FunctionKeys) {
  auto apply = [&](const KeyTable &Keys, PackIndexView::Table T,
                   std::vector<uint32_t> &Ids, uint32_t Count) {
    Ids.assign(Count, NoIndex);
    std::vector<std::pair<RecordKey, uint32_t>> Sorted;
    Keys.forEach([&](const RecordKey &Key, uint32_t Id) {
      Sorted.push_back({Key, Id});
    });
    llvm::sort(Sorted,
               [](const auto &A, const auto &B) { return A.first < B.first; });
    std::vector<RecordKey> SortedKeys;
    SortedKeys.reserve(Sorted.size());
    for (const auto &Entry : Sorted)
      SortedKeys.push_back(Entry.first);
    Index.findSorted(T, SortedKeys,
                     [&](size_t K, uint32_t I) { Ids[I] = Sorted[K].second; });
  };
  apply(TypeKeys, PackIndexView::Table::Types, E.KeyIds, Index.typeCount());
  apply(SymbolKeys, PackIndexView::Table::Symbols, E.SymbolKeyIds,
        Index.symbolCount());
  apply(FunctionKeys, PackIndexView::Table::Functions, E.FunctionKeyIds,
        Index.functionCount());
}

namespace {
/// A function of the new pack, copied from its owner object.
struct FunctionEntry {
  uint32_t Object = 0;
  uint32_t Function = 0;
};

/// A file of the new pack, copied from its owner object.
struct FileEntry {
  uint32_t Object = 0;
  uint32_t KeyId = 0;
};

struct OwnedFile {
  std::string Name;
  uint8_t ChecksumKind = 0;
  std::vector<uint8_t> Checksum;
};

/// A type record of the new pack, copied from its owner object.
struct PackEntry {
  uint32_t Object = 0;
  uint32_t Record = 0;
};

struct SymbolEntry {
  uint32_t Object = 0;
  uint32_t Candidate = 0;
};
} // namespace

/// Encode a range map from a sequence of (pack index or 0) per record.
static RangeMap buildRangeMap(ArrayRef<uint32_t> Mapped, uint64_t FirstIndex) {
  RangeMap Map;
  uint64_t Skip = 0;
  uint64_t PreviousEnd = FirstIndex;
  for (size_t I = 0; I != Mapped.size();) {
    if (Mapped[I] == 0) {
      ++Skip;
      ++I;
      continue;
    }
    uint64_t First = Mapped[I];
    uint32_t Count = 1;
    ++I;
    while (I != Mapped.size() && Mapped[I] != 0 && Mapped[I] == First + Count) {
      ++Count;
      ++I;
    }
    Map.Entries.push_back(
        {Skip, 0, Count, int64_t(First) - int64_t(PreviousEnd)});
    PreviousEnd = First + Count;
    Skip = 0;
  }
  return Map;
}

namespace {
struct StageResult {
  std::vector<uint8_t> Data;
  bool Packed = false;
  uint64_t MapEntries = 0;
  uint64_t MapBytes = 0;
  uint64_t Functions = 0;
  uint64_t FunctionBytes = 0;
};
} // namespace

/// Build the packed form of one object.
static Expected<StageResult> stageObject(const InputObject &Object,
                                         const ObjectView &View,
                                         const PackId &PackIdValue,
                                         CVPackResolver &Resolver) {
  StageResult Result;
  ArrayRef<uint8_t> Bytes = View.Bytes;
  if (llvm::none_of(Object.Mapped, [](uint32_t M) { return M; })) {
    Result.Data.assign(Bytes.begin(), Bytes.end());
    return Result;
  }

  Origin OriginData;
  OriginData.OriginalFileSize = Bytes.size();
  OriginData.OriginalFileHash = BLAKE3::hash(Bytes);
  std::vector<SectionReplacement> Replacements;

  // .debug$T
  ArrayRef<uint8_t> OriginalTypes = View.section(Object.TypeSection);
  auto ListRecord = encodePackList(PackIdValue);
  if (!ListRecord)
    return ListRecord.takeError();
  RangeMap TypeMap =
      buildRangeMap(Object.Mapped, TypeIndex::FirstNonSimpleIndex);
  auto MapRecords = encodeTypeMap(TypeMap);
  if (!MapRecords)
    return MapRecords.takeError();
  Result.MapEntries = TypeMap.Entries.size();
  std::vector<uint8_t> TypeBody;
  for (const auto &Record : *MapRecords) {
    TypeBody.insert(TypeBody.end(), Record.begin(), Record.end());
    Result.MapBytes += Record.size();
  }
  for (size_t I = 0; I != View.Types.size(); ++I)
    if (!Object.Mapped[I])
      TypeBody.insert(TypeBody.end(), View.Types[I].begin(),
                      View.Types[I].end());
  OriginEdit TypeEdit;
  TypeEdit.SectionIndex = Object.TypeSection;
  TypeEdit.Kind = OriginEditKind::TypeStream;
  TypeEdit.OriginalLength = OriginalTypes.size();
  OriginData.Edits.push_back(TypeEdit);

  // .debug$H
  if (Object.DropDebugH) {
    ArrayRef<uint8_t> HashData = View.section(Object.HashSection);
    OriginData.DebugHHeader.assign(HashData.begin(), HashData.begin() + 8);
    OriginEdit Edit;
    Edit.SectionIndex = Object.HashSection;
    Edit.Kind = OriginEditKind::DebugH;
    Edit.OriginalLength = HashData.size();
    Edit.PackedLength = 8;
    // One fragment per run of consecutive differing hashes.
    for (size_t I = 0; I != Object.DebugHPatches.size();) {
      size_t J = I + 1;
      while (J != Object.DebugHPatches.size() &&
             Object.DebugHPatches[J] == Object.DebugHPatches[J - 1] + 1)
        ++J;
      uint32_t Offset = 8 + 8 * Object.DebugHPatches[I];
      RestorationFragment Fragment;
      Fragment.Offset = Offset;
      Fragment.Bytes.assign(HashData.begin() + Offset,
                            HashData.begin() + Offset + 8 * (J - I));
      Edit.RestorationFragments.push_back(std::move(Fragment));
      I = J;
    }
    OriginData.Edits.push_back(Edit);
    SectionReplacement R;
    R.SectionIndex = Object.HashSection;
    R.Data = OriginData.DebugHHeader;
    Replacements.push_back(std::move(R));
  }

  // Symbol subsections, grouped by section.
  std::map<uint32_t, std::vector<uint32_t>> SubsectionsBySection;
  for (uint32_t S = 0; S != Object.Subsections.size(); ++S)
    SubsectionsBySection[Object.Subsections[S].Section].push_back(S);
  for (auto &[Section, Indices] : SubsectionsBySection) {
    ArrayRef<uint8_t> Data = View.section(Section);
    SectionReplacement R;
    R.SectionIndex = Section;
    size_t Cursor = 0;
    auto keep = [&](size_t From, size_t Size) {
      if (Size != 0)
        R.Spans.push_back({static_cast<uint32_t>(From),
                           static_cast<uint32_t>(R.Data.size()),
                           static_cast<uint32_t>(Size)});
      R.Data.insert(R.Data.end(), Data.begin() + From,
                    Data.begin() + From + Size);
    };
    bool Changed = false;
    for (uint32_t S : Indices) {
      const SymbolSubsection &Subsection = Object.Subsections[S];
      // Per-record pack ordinal + 1, or 0 for inline symbols.
      std::vector<uint32_t> Mapped;
      uint64_t Removed = 0;
      for (const SymbolRecordRef &Record : Subsection.Records) {
        uint32_t Ordinal = Record.Candidate == NoIndex
                               ? NoIndex
                               : Object.SymbolMapped[Record.Candidate];
        Mapped.push_back(Ordinal == NoIndex ? 0 : Ordinal + 1);
        if (Ordinal != NoIndex)
          Removed += Record.Size;
      }
      if (Removed == 0)
        continue;
      auto Encoded = encodeSymbolMap(buildRangeMap(Mapped, 1));
      if (!Encoded)
        return Encoded.takeError();
      uint64_t Inserted = 0;
      for (const auto &Record : *Encoded)
        Inserted += Record.size();
      if (Inserted >= Removed || (Removed - Inserted) % 4 != 0)
        continue;

      keep(Cursor, Subsection.HeaderOffset - Cursor);
      size_t NewHeader = R.Data.size();
      R.Data.insert(R.Data.end(), Data.begin() + Subsection.HeaderOffset,
                    Data.begin() + Subsection.HeaderOffset + 8);
      for (const auto &Record : *Encoded)
        R.Data.insert(R.Data.end(), Record.begin(), Record.end());
      for (size_t I = 0; I != Subsection.Records.size(); ++I)
        if (!Mapped[I])
          keep(Subsection.Records[I].Offset, Subsection.Records[I].Size);
      uint32_t NewSize = R.Data.size() - NewHeader - 8;
      support::endian::write32le(R.Data.data() + NewHeader + 4, NewSize);

      OriginEdit Edit;
      Edit.SectionIndex = Section;
      Edit.Kind = OriginEditKind::SymbolStream;
      Edit.OriginalOffset = Subsection.HeaderOffset;
      Edit.OriginalLength = 8 + Subsection.DataSize;
      Edit.PackedLength = 8 + NewSize;
      OriginData.Edits.push_back(Edit);
      Cursor = Subsection.HeaderOffset + 8 + Subsection.DataSize;
      Changed = true;
    }
    if (!Changed)
      continue;
    keep(Cursor, Data.size() - Cursor);
    Replacements.push_back(std::move(R));
  }

  // Function debug info moved to the pack: the section keeps a reference and
  // its relocations, at their original offsets.
  for (uint32_t F = 0; F != Object.FunctionMapped.size(); ++F) {
    uint32_t Ordinal = Object.FunctionMapped[F];
    if (Ordinal == NoIndex)
      continue;
    if (Ordinal == Selected)
      return producerError("function selected without a pack ordinal");
    const FunctionDebug &Function = Object.Functions[F];
    SectionReplacement R;
    R.SectionIndex = Function.Section;
    R.Data = encodeFunctionRef({0, Ordinal, Function.Size});
    R.Spans.push_back({0, 0, Function.Size});
    Replacements.push_back(std::move(R));
    ++Result.Functions;
    Result.FunctionBytes += Function.Size;
  }

  // The directive that makes linkers without CodeView pack support fail with
  // an explicit error. It goes into the object's .drectve, before any trailing
  // NUL bytes; an object without one gets a new last section. Only one
  // .drectve section is ever present: LLD reads the last one.
  ArrayRef<uint8_t> Directive = arrayRefFromStringRef(PackedObjectDirective);
  uint32_t DirectiveSection = 0;
  for (uint32_t I = 1; I <= View.Sections.size(); ++I)
    if (View.Sections.name(I) == ".drectve")
      DirectiveSection = I;
  bool AddDirectiveSection = DirectiveSection == 0;
  {
    OriginEdit Edit;
    Edit.PackedLength = Directive.size();
    if (AddDirectiveSection) {
      Edit.SectionIndex = View.Sections.size() + 1;
      Edit.Kind = OriginEditKind::AddedDirectiveSection;
    } else {
      ArrayRef<uint8_t> Data = View.section(DirectiveSection);
      size_t At = Data.size();
      while (At && Data[At - 1] == 0)
        --At;
      Edit.SectionIndex = DirectiveSection;
      Edit.Kind = OriginEditKind::Directive;
      Edit.OriginalOffset = At;
      SectionReplacement R;
      R.SectionIndex = DirectiveSection;
      R.Data.assign(Data.begin(), Data.begin() + At);
      R.Data.insert(R.Data.end(), Directive.begin(), Directive.end());
      R.Data.insert(R.Data.end(), Data.begin() + At, Data.end());
      if (At)
        R.Spans.push_back({0, 0, static_cast<uint32_t>(At)});
      if (At != Data.size())
        R.Spans.push_back({static_cast<uint32_t>(At),
                           static_cast<uint32_t>(At + Directive.size()),
                           static_cast<uint32_t>(Data.size() - At)});
      Replacements.push_back(std::move(R));
    }
    OriginData.Edits.push_back(Edit);
  }

  llvm::sort(OriginData.Edits, [](const OriginEdit &A, const OriginEdit &B) {
    return std::make_pair(A.SectionIndex, A.OriginalOffset) <
           std::make_pair(B.SectionIndex, B.OriginalOffset);
  });

  // The type stream's packed length includes the origin record itself, so
  // encode the origin twice.
  auto encodeTypeSection = [&]() -> Expected<std::vector<uint8_t>> {
    auto OriginRecord = encodeOrigin(OriginData);
    if (!OriginRecord)
      return OriginRecord.takeError();
    std::vector<uint8_t> Section;
    appendLE<uint32_t>(Section, COFF::DEBUG_CVPACK_SECTION_MAGIC);
    Section.insert(Section.end(), ListRecord->begin(), ListRecord->end());
    Section.insert(Section.end(), OriginRecord->begin(), OriginRecord->end());
    Section.insert(Section.end(), TypeBody.begin(), TypeBody.end());
    return Section;
  };
  auto TypeSectionData = encodeTypeSection();
  if (!TypeSectionData)
    return TypeSectionData.takeError();
  for (OriginEdit &Edit : OriginData.Edits)
    if (Edit.Kind == OriginEditKind::TypeStream)
      Edit.PackedLength = TypeSectionData->size();
  TypeSectionData = encodeTypeSection();
  if (!TypeSectionData)
    return TypeSectionData.takeError();
  SectionReplacement TypeReplacement;
  TypeReplacement.SectionIndex = Object.TypeSection;
  TypeReplacement.Data = std::move(*TypeSectionData);
  Replacements.push_back(std::move(TypeReplacement));

  auto Staged = replaceCOFFSections(Bytes, Replacements);
  if (!Staged)
    return Staged.takeError();
  if (AddDirectiveSection) {
    Staged = appendCOFFSection(*Staged, ".drectve",
                               COFF::IMAGE_SCN_LNK_INFO |
                                   COFF::IMAGE_SCN_LNK_REMOVE |
                                   COFF::IMAGE_SCN_ALIGN_1BYTES,
                               Directive);
    if (!Staged)
      return Staged.takeError();
  }

  auto Restored = materializeObject(*Staged, Resolver);
  if (!Restored)
    return Restored.takeError();
  if (ArrayRef<uint8_t>(*Restored) != Bytes)
    return producerError("staged object does not materialize byte-identically");
  Result.Data = std::move(*Staged);
  Result.Packed = true;
  return Result;
}

/// Write through a temporary file, renamed when complete: a killed or
/// concurrent run never leaves a truncated file under the final name.
static Error writeFile(StringRef Path, ArrayRef<uint8_t> Data) {
  return writeToOutput(Path, [&](raw_ostream &OS) {
    OS << toStringRef(Data);
    return Error::success();
  });
}

namespace {
/// What a run reports to its caller.
struct RunOutcome {
  /// Set when a run against a base found that it no longer fits; nothing
  /// was written and the caller starts over as a rebase.
  std::string RebaseReason;
  /// Every pack that some staged object needs.
  std::vector<std::string> Required;
};

/// The state of one run, which its phases share.
struct ProducerRun {
  ProducerRun(const PackOptions &Options, bool Rebase, StringRef RebasedBecause,
              ArrayRef<StringRef> PackDirs)
      : Options(Options), StagingDirectory(Options.StagingDirectory),
        OutputPackDir(Options.PackPaths.front()), Rebase(Rebase),
        RebasedBecause(RebasedBecause), Resolver(PackDirs),
        Start(std::chrono::steady_clock::now()) {}

  /// Log the time elapsed since the run started.
  void phase(const char *Name) const {
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Start;
    Message(Options.Log) << format("%7.1f", Elapsed.count()) << " s " << Name;
  }
  /// The index at which the base holds a key, if an object can refer to it.
  uint32_t baseIndex(uint32_t Key) const {
    uint32_t I = BaseLoc[Key];
    return I != NoIndex && BaseImportable[I] ? I : NoIndex;
  }
  /// Whether the records of a key are packed when everything they reference
  /// is: those the base holds, or on a rebase those of several objects.
  bool worthPacking(uint32_t Key) const {
    return BaseMode ? baseIndex(Key) != NoIndex : TypeCount.shared(Key);
  }

  const PackOptions &Options;
  StringRef StagingDirectory;
  StringRef OutputPackDir;
  bool Rebase;
  StringRef RebasedBecause;
  // One resolver for the whole run: a pack is loaded and expanded once, both
  // to materialize packed inputs and to verify staged objects.
  CVPackResolver Resolver;
  std::chrono::steady_clock::time_point Start;
  // In-flight input bytes per parallel phase; the buffers of a phase are a
  // few times the size of its inputs.
  MemoryBudget Budget{uint64_t(1) << 30};

  // The inputs, in path order, and the errors that each one hit.
  std::vector<InputObject> Objects;
  std::vector<std::string> Errors;
  std::unique_ptr<KeyTable> TypeKeys, SymbolKeys, FunctionKeys, FileKeys;
  uint16_t Machine = COFF::IMAGE_FILE_MACHINE_UNKNOWN;

  // The dedup set, and the objects containing each key.
  std::vector<ExistingPack> Existing;
  size_t DedupSetSize = 0;
  uint32_t KeyCount = 0, SymbolKeyCount = 0, FunctionKeyCount = 0;
  SharedCounter TypeCount, SymbolCount, FunctionCount;

  // The base, and how the inputs use it.
  std::optional<ExistingPack> Base;
  bool BaseMode = false;
  std::vector<uint32_t> BaseLoc; // key -> base array index
  std::vector<uint8_t> BaseImportable;
  DenseMap<uint32_t, uint32_t> BaseSymbols;   // key -> symbol ordinal
  DenseMap<uint32_t, uint32_t> BaseFunctions; // key -> function ordinal
  std::vector<std::atomic<uint8_t>> BaseTypeUsed, BaseFunctionUsed;
  uint64_t BaseBytes = 0, BaseUsedBytes = 0;
  // Base mode: bytes that a rebase would move out of the objects, less those
  // that the base moves.
  std::atomic<int64_t> MissingBytes{0};

  // The records of the new pack, each copied from its owner object.
  std::vector<PackEntry> Entries;
  std::vector<SymbolEntry> SymbolEntries;
  std::vector<FunctionEntry> FunctionEntries;
  std::vector<FileEntry> FileEntries;
  std::vector<uint32_t> FileOrdinal; // file key -> pack file ordinal
  bool NeedNewPack = false;
  std::optional<PackId> NewPackId;
  uint64_t PackBytes = 0;

  // The staged objects.
  std::vector<std::string> StagedNames;
  std::vector<uint8_t> PackedFlags;
  std::atomic<uint64_t> InputBytes{0}, OutputBytes{0}, Fallbacks{0};
  std::atomic<uint64_t> MapEntries{0}, MapBytes{0};
  std::atomic<uint64_t> StagedFunctions{0}, StagedFunctionBytes{0};
};

/// The records of the new pack, copied from their owners, and the records
/// that each object owns.
struct NewPackRecords {
  explicit NewPackRecords(const ProducerRun &Run)
      : Types(Run.Entries.size()), Symbols(Run.SymbolEntries.size()),
        Functions(Run.FunctionEntries.size()), Files(Run.FileEntries.size()),
        TypesOwnedBy(Run.Objects.size()), SymbolsOwnedBy(Run.Objects.size()),
        FunctionsOwnedBy(Run.Objects.size()), FilesOwnedBy(Run.Objects.size()) {
    for (uint32_t F = 0; F != Run.FunctionEntries.size(); ++F)
      FunctionsOwnedBy[Run.FunctionEntries[F].Object].push_back(F);
    for (uint32_t F = 0; F != Run.FileEntries.size(); ++F)
      FilesOwnedBy[Run.FileEntries[F].Object].push_back(F);
    for (uint32_t E = 0; E != Run.Entries.size(); ++E)
      TypesOwnedBy[Run.Entries[E].Object].push_back(E);
    for (uint32_t S = 0; S != Run.SymbolEntries.size(); ++S)
      SymbolsOwnedBy[Run.SymbolEntries[S].Object].push_back(S);
  }

  std::vector<std::vector<uint8_t>> Types;
  std::vector<std::vector<uint8_t>> Symbols;
  std::vector<std::vector<uint8_t>> Functions;
  std::vector<OwnedFile> Files;
  std::vector<std::vector<uint32_t>> TypesOwnedBy;
  std::vector<std::vector<uint32_t>> SymbolsOwnedBy;
  std::vector<std::vector<uint32_t>> FunctionsOwnedBy;
  std::vector<std::vector<uint32_t>> FilesOwnedBy;
};
} // namespace

/// Phase 1: read and key every object in parallel.
static void analyzeInputs(ProducerRun &Run, ArrayRef<StringRef> InputPaths) {
  std::vector<std::string> SortedPaths;
  for (StringRef Path : InputPaths)
    SortedPaths.push_back(Path.str());
  llvm::sort(SortedPaths);
  SortedPaths.erase(llvm::unique(SortedPaths), SortedPaths.end());

  Run.Objects.resize(SortedPaths.size());
  Run.Errors.resize(SortedPaths.size());
  for (size_t I = 0; I != Run.Objects.size(); ++I)
    Run.Objects[I].Path = SortedPaths[I];
  Run.TypeKeys = std::make_unique<KeyTable>();
  Run.SymbolKeys = std::make_unique<KeyTable>();
  Run.FunctionKeys = std::make_unique<KeyTable>();
  Run.FileKeys = std::make_unique<KeyTable>();
  parallelFor(0, Run.Objects.size(), [&](size_t I) {
    uint64_t Size = 0;
    sys::fs::file_size(Run.Objects[I].Path, Size);
    Run.Budget.acquire(Size);
    analyzeObject(Run.Objects[I], Run.Resolver, *Run.TypeKeys, *Run.SymbolKeys,
                  *Run.FunctionKeys, *Run.FileKeys);
    Run.Budget.release(Size);
  });
}

/// A pack holds the records of the objects of one machine: that of the most
/// inputs. The inputs of other machines are staged unchanged.
static void chooseMachine(ProducerRun &Run) {
  DenseMap<uint16_t, size_t> Inputs;
  size_t Most = 0;
  for (const InputObject &Object : Run.Objects) {
    if (!Object.SkipReason.empty())
      continue;
    size_t Count = ++Inputs[Object.Machine];
    if (Count > Most) {
      Most = Count;
      Run.Machine = Object.Machine;
    }
  }
  for (InputObject &Object : Run.Objects)
    if (Object.SkipReason.empty() && Object.Machine != Run.Machine)
      Object.SkipReason = "machine differs from that of the other inputs";
}

/// The packs of the dedup set: the base that the options name, or those of the
/// pack directories.
static std::vector<PackId> dedupSetIds(const PackOptions &Options) {
  std::vector<PackId> Ids;
  if (Options.Base) {
    Ids.push_back(*Options.Base);
    return Ids;
  }
  std::set<std::string> Seen;
  for (const std::string &Dir : Options.PackPaths) {
    std::error_code EC;
    for (sys::fs::directory_iterator It(Dir, EC), End; It != End && !EC;
         It.increment(EC)) {
      StringRef Name = sys::path::filename(It->path());
      if (!Name.ends_with_insensitive(PackFileExtension))
        continue;
      auto Id = parseIdString(Name.drop_back(PackFileExtension.size()));
      if (Id && Seen.insert(idString(*Id)).second)
        Ids.push_back(*Id);
    }
  }
  llvm::sort(Ids);
  return Ids;
}

/// The index of a pack, found next to it in the pack directories, if one
/// describes the pack. The returned buffer holds the view's data.
static std::unique_ptr<MemoryBuffer>
findPackIndex(const ProducerRun &Run, const PackId &Id,
              std::optional<PackIndexView> &View) {
  const PackOptions &Options = Run.Options;
  std::string PackName = idString(Id) + PackFileExtension.str();
  std::string IndexName = idString(Id) + PackIndexFileExtension.str();
  uint64_t PackSize = 0;
  bool PackFound = false;
  for (const std::string &Dir : Options.PackPaths) {
    SmallString<256> Path(Dir);
    sys::path::append(Path, PackName);
    if (!sys::fs::file_size(Path, PackSize)) {
      PackFound = true;
      break;
    }
  }
  if (!PackFound)
    return nullptr;
  for (const std::string &Dir : Options.PackPaths) {
    SmallString<256> Path(Dir);
    sys::path::append(Path, IndexName);
    // Read rather than mapped: lookups touch pages all over the file, and
    // one sequential read is much cheaper than faulting each of them in.
    auto Buffer = MemoryBuffer::getFile(Path, /*IsText=*/false,
                                        /*RequiresNullTerminator=*/false,
                                        /*IsVolatile=*/true);
    if (!Buffer)
      continue;
    auto Candidate =
        PackIndexView::create(arrayRefFromStringRef((*Buffer)->getBuffer()));
    if (!Candidate) {
      Message(Run.Options.Log)
          << "ignoring " << Path << ": " << toString(Candidate.takeError());
      continue;
    }
    if (Candidate->packContentId() != Id ||
        Candidate->packFileSize() != PackSize) {
      Message(Run.Options.Log)
          << "ignoring " << Path << ": it does not describe the pack";
      continue;
    }
    View = *Candidate;
    return std::move(*Buffer);
  }
  return nullptr;
}

/// Load a pack of the dedup set and key it through its index, which is
/// built, and kept in the output directory, when it has none.
static Error loadExistingPack(ProducerRun &Run, const PackId &Id,
                              uint64_t &IndexesLoaded, uint64_t &IndexesBuilt) {
  const PackOptions &Options = Run.Options;
  std::string PackName = idString(Id) + PackFileExtension.str();
  // The index is found first: a pack described by its index is loaded
  // without hashing it, since its index was built from the checked pack and
  // every staged object is verified by restoring it.
  std::optional<PackIndexView> View;
  std::unique_ptr<MemoryBuffer> IndexBuffer = findPackIndex(Run, Id, View);
  // A pack of the directories that does not load, such as a file that
  // another program truncated, is ignored. A base that the options name must
  // load.
  auto unusable = [&](Error E) -> Error {
    if (Options.Base)
      return E;
    Message(Options.Warn) << "ignoring " << PackName << ": "
                          << toString(std::move(E));
    return Error::success();
  };
  auto Loaded = Run.Resolver.get(Id, /*VerifyContentId=*/!View);
  if (!Loaded)
    return unusable(Loaded.takeError());
  if ((*Loaded)->View.Machine != Run.Machine) {
    if (Options.Base)
      return producerError("CodeView pack " + PackName +
                           " is for another machine than the inputs");
    Message(Options.Log) << "ignoring " << PackName
                         << ": it is for another machine";
    return Error::success();
  }
  Run.phase(View ? "base loaded, trusted" : "base loaded");
  ExistingPack E;
  E.Pack = *Loaded;
  E.Id = Id;
  if (View && (View->typeCount() != E.Pack->View.Types.size() ||
               View->symbolCount() != E.Pack->View.Symbols.size() ||
               View->functionCount() != E.Pack->View.Functions.size())) {
    Message(Options.Log) << "ignoring the index of " << PackName
                         << ": it does not describe the pack";
    View.reset();
    ArrayRef<uint8_t> Data = arrayRefFromStringRef(E.Pack->Buffer->getBuffer());
    if (BLAKE3::hash(Data) != Id)
      return unusable(producerError("CodeView pack " + PackName +
                                    " does not match its content id"));
  }
  std::vector<uint8_t> Built;
  if (View) {
    ++IndexesLoaded;
  } else {
    auto Index = buildPackIndex(*E.Pack, Id);
    if (!Index)
      return Index.takeError();
    auto Encoded = writePackIndex(std::move(*Index));
    if (!Encoded)
      return Encoded.takeError();
    Built = std::move(*Encoded);
    View = cantFail(PackIndexView::create(Built));
    ++IndexesBuilt;
    // Keep the index next to the pack when the pack is in the output
    // directory, so that later runs need not key the pack again.
    SmallString<256> PackPath(Run.OutputPackDir);
    sys::path::append(PackPath, PackName);
    if (sys::fs::exists(PackPath)) {
      SmallString<256> IndexPath(Run.OutputPackDir);
      sys::path::append(IndexPath, idString(Id) + PackIndexFileExtension.str());
      if (Error Err = writeFile(IndexPath, Built))
        return Err;
    }
  }
  applyPackIndex(*View, E, *Run.TypeKeys, *Run.SymbolKeys, *Run.FunctionKeys);
  Run.phase("index applied");
  Run.Existing.push_back(std::move(E));
  return Error::success();
}

/// Phase 2: load the dedup set, unless the run is a rebase.
static Error loadDedupSet(ProducerRun &Run) {
  if (Run.Rebase)
    return Error::success();
  uint64_t IndexesLoaded = 0, IndexesBuilt = 0;
  for (const PackId &Id : dedupSetIds(Run.Options))
    if (Error E = loadExistingPack(Run, Id, IndexesLoaded, IndexesBuilt))
      return E;
  if (IndexesLoaded || IndexesBuilt)
    Message(Run.Options.Log) << IndexesLoaded << " pack indexes loaded, "
                             << IndexesBuilt << " built";
  return Error::success();
}

/// Every key is interned; only the dense ids are used from here on. File
/// keys are few, and owners look theirs up again when the pack is written.
static Error finishKeys(ProducerRun &Run) {
  Run.KeyCount = Run.TypeKeys->size();
  Run.SymbolKeyCount = Run.SymbolKeys->size();
  Run.FunctionKeyCount = Run.FunctionKeys->size();
  if (Run.KeyCount >= NoKey)
    return producerError("too many distinct type records");
  Run.TypeKeys.reset();
  Run.SymbolKeys.reset();
  Run.FunctionKeys.reset();
  return Error::success();
}

/// Count the objects containing each key, saturating at 2. A key repeated
/// within one object counts twice, which at worst packs a record that only
/// one object uses.
static void countKeys(ProducerRun &Run) {
  Run.TypeCount.resize(Run.KeyCount);
  Run.SymbolCount.resize(Run.SymbolKeyCount);
  Run.FunctionCount.resize(Run.FunctionKeyCount);
  parallelFor(0, Run.Objects.size(), [&](size_t O) {
    const InputObject &Object = Run.Objects[O];
    if (!Object.SkipReason.empty())
      return;
    for (uint32_t Key : Object.Mapped)
      if (Key != NoKey)
        Run.TypeCount.add(Key);
    for (const SymbolCandidate &Symbol : Object.Symbols)
      Run.SymbolCount.add(Symbol.KeyId);
    for (const FunctionDebug &Function : Object.Functions)
      Run.FunctionCount.add(Function.KeyId);
  });
}

/// The base is the existing pack that holds the most bytes of records the
/// inputs contain; the others are not referenced. Without an existing pack,
/// the run writes the first base.
static void chooseBase(ProducerRun &Run) {
  Run.DedupSetSize = Run.Existing.size();
  if (!Run.Existing.empty()) {
    uint64_t Best = 0;
    size_t BestIndex = 0;
    for (size_t P = 0; P != Run.Existing.size(); ++P) {
      const ExistingPack &E = Run.Existing[P];
      uint64_t Covered = 0;
      for (size_t I = 0; I != E.KeyIds.size(); ++I)
        if (E.KeyIds[I] != NoIndex && Run.TypeCount.present(E.KeyIds[I]))
          Covered += E.Pack->View.Types[I].length();
      for (size_t F = 0; F != E.FunctionKeyIds.size(); ++F)
        if (E.FunctionKeyIds[F] != NoIndex &&
            Run.FunctionCount.present(E.FunctionKeyIds[F]))
          Covered += E.Pack->View.Functions[F].size();
      if (P == 0 || Covered > Best) {
        Best = Covered;
        BestIndex = P;
      }
    }
    Run.Base = std::move(Run.Existing[BestIndex]);
    Run.Existing.clear();
  }
  Run.BaseMode = Run.Base.has_value();

  // A packed input whose pack is the base, typically a cache hit restored on
  // the producer, is staged unchanged. Its records still count towards the
  // sharing decisions for the other inputs.
  for (InputObject &Object : Run.Objects) {
    if (Run.BaseMode && Object.SkipReason.empty() &&
        Object.InputPack == Run.Base->Id) {
      Object.PassThrough = true;
      Object.Materialized = {};
    }
  }
}

/// Against the base, a record is only packed when the base holds it at the
/// first index of its key and every record it references is held the same
/// way: the object then references the base, and no new pack is needed.
static void indexBase(ProducerRun &Run) {
  Run.BaseLoc.assign(Run.KeyCount, NoIndex);
  if (!Run.BaseMode)
    return;
  const ExistingPack &E = *Run.Base;
  for (uint32_t I = 0; I != E.KeyIds.size(); ++I)
    if (E.KeyIds[I] != NoIndex && Run.BaseLoc[E.KeyIds[I]] == NoIndex)
      Run.BaseLoc[E.KeyIds[I]] = I;
  Run.BaseImportable.assign(E.KeyIds.size(), 0);
  SmallVector<uint32_t, 16> Refs;
  for (uint32_t I = 0; I != E.KeyIds.size(); ++I) {
    ArrayRef<uint8_t> Record = E.Pack->View.Types[I].RecordData;
    Run.BaseBytes += Record.size();
    uint32_t Key = E.KeyIds[I];
    if (Key == NoIndex || Run.BaseLoc[Key] != I)
      continue;
    Refs.clear();
    if (!collectRefs(Record, false, I, Refs))
      continue;
    Run.BaseImportable[I] = llvm::all_of(
        Refs, [&](uint32_t R) { return Run.BaseImportable[R] != 0; });
  }
  for (uint32_t S = 0; S != E.SymbolKeyIds.size(); ++S)
    if (E.SymbolKeyIds[S] != NoIndex)
      Run.BaseSymbols.try_emplace(E.SymbolKeyIds[S], S);
  for (uint32_t F = 0; F != E.FunctionKeyIds.size(); ++F) {
    Run.BaseBytes += E.Pack->View.Functions[F].size();
    if (E.FunctionKeyIds[F] != NoIndex)
      Run.BaseFunctions.try_emplace(E.FunctionKeyIds[F], F);
  }
  Run.BaseTypeUsed = std::vector<std::atomic<uint8_t>>(E.KeyIds.size());
  Run.BaseFunctionUsed =
      std::vector<std::atomic<uint8_t>>(E.FunctionKeyIds.size());
}

/// Select the records of an object to pack. A record is selected when it is
/// worth packing, not a repeat within its object, and everything it
/// references is selected.
static void selectObjectRecords(ProducerRun &Run, InputObject &Object) {
  const bool BaseMode = Run.BaseMode;
  // In base mode, packed inputs staged unchanged are selected too, only to
  // measure how they use the base: their packing depends on nothing else.
  bool MeasureOnly =
      BaseMode && Object.SkipReason.empty() && Object.PassThrough;
  if (!Object.packable() && !MeasureOnly)
    return;
  Run.Budget.acquire(Object.Size);
  scope_exit Release([&] { Run.Budget.release(Object.Size); });
  ObjectView View;
  if (Error E = openView(Object, View, &Run.Resolver)) {
    if (!MeasureOnly)
      Object.SkipReason = toString(std::move(E));
    else
      consumeError(std::move(E));
    return;
  }
  auto isSelected = [&](uint32_t R) { return Object.Mapped[R] & SelectedBit; };
  DenseSet<uint32_t> Used;
  SmallVector<uint32_t, 16> Refs;
  // What a rebase would select: shared records, closed under references.
  int64_t Missing = 0;
  std::vector<uint8_t> RebaseSelected;
  if (BaseMode) {
    RebaseSelected.assign(Object.Mapped.size(), 0);
    DenseSet<uint32_t> UsedByRebase;
    for (uint32_t I = 0; I != Object.Mapped.size(); ++I) {
      uint32_t Key = Object.Mapped[I];
      if (Key == NoKey || !Run.TypeCount.shared(Key) ||
          UsedByRebase.contains(Key))
        continue;
      Refs.clear();
      if (!collectRefs(View.Types[I], false, I, Refs) ||
          !llvm::all_of(Refs, [&](uint32_t R) { return RebaseSelected[R]; }))
        continue;
      RebaseSelected[I] = 1;
      UsedByRebase.insert(Key);
      Missing += View.Types[I].size();
    }
  }
  for (uint32_t I = 0; I != Object.Mapped.size(); ++I) {
    uint32_t Key = Object.Mapped[I];
    if (Key == NoKey || !Run.worthPacking(Key) || Used.contains(Key))
      continue;
    Refs.clear();
    if (!collectRefs(View.Types[I], false, I, Refs))
      continue;
    if (!llvm::all_of(Refs, isSelected))
      continue;
    Object.Mapped[I] |= SelectedBit;
    Used.insert(Key);
    if (BaseMode) {
      Missing -= View.Types[I].size();
      Run.BaseTypeUsed[Run.baseIndex(Key)].store(1, std::memory_order_relaxed);
    }
  }
  Object.SymbolMapped.assign(Object.Symbols.size(), NoIndex);
  DenseSet<uint32_t> UsedSymbols;
  for (uint32_t S = 0; S != Object.Symbols.size(); ++S) {
    const SymbolCandidate &Symbol = Object.Symbols[S];
    bool Worth = BaseMode ? Run.BaseSymbols.contains(Symbol.KeyId)
                          : Run.SymbolCount.shared(Symbol.KeyId);
    if (!Worth || UsedSymbols.contains(Symbol.KeyId))
      continue;
    if (!llvm::all_of(Symbol.Refs, isSelected))
      continue;
    Object.SymbolMapped[S] = Selected;
    UsedSymbols.insert(Symbol.KeyId);
  }
  // A function is packed when it is shared and every type it references is
  // packed. Each referenced record is then the first of its key in the
  // object, so that the pack index maps back to it.
  Object.FunctionMapped.assign(Object.Functions.size(), NoIndex);
  for (uint32_t F = 0; F != Object.Functions.size(); ++F) {
    const FunctionDebug &Function = Object.Functions[F];
    int64_t Moved = int64_t(Function.Size) - int64_t(FunctionRefSectionSize);
    ArrayRef<uint32_t> Types =
        ArrayRef<uint32_t>(Object.FunctionTypes)
            .slice(Function.FirstType, Function.NumTypes);
    if (BaseMode && Run.FunctionCount.shared(Function.KeyId) &&
        llvm::all_of(Types, [&](uint32_t R) { return RebaseSelected[R]; }))
      Missing += Moved;
    bool Worth = BaseMode ? Run.BaseFunctions.contains(Function.KeyId)
                          : Run.FunctionCount.shared(Function.KeyId);
    if (!Worth) {
      auto &Entry =
          Object.FunctionSkips[BaseMode ? "not in the base" : "not shared"];
      ++Entry.first;
      Entry.second += Function.Size;
      continue;
    }
    if (!llvm::all_of(Types, isSelected)) {
      auto &Entry = Object.FunctionSkips["references an unpacked type"];
      ++Entry.first;
      Entry.second += Function.Size;
      continue;
    }
    Object.FunctionMapped[F] = Selected;
    if (BaseMode) {
      Missing -= Moved;
      Run.BaseFunctionUsed[Run.BaseFunctions.lookup(Function.KeyId)].store(
          1, std::memory_order_relaxed);
    }
  }
  Run.MissingBytes += Missing;
}

/// Phase 3a: select the records to pack, in parallel.
static void selectRecords(ProducerRun &Run) {
  parallelFor(0, Run.Objects.size(),
              [&](size_t O) { selectObjectRecords(Run, Run.Objects[O]); });
}

/// Base mode: decide whether the base still fits the inputs. Returns why it
/// does not, or an empty string.
static std::string checkBaseFit(ProducerRun &Run) {
  if (!Run.BaseMode)
    return {};
  const PackOptions &Options = Run.Options;
  const ExistingPack &E = *Run.Base;
  for (size_t I = 0; I != Run.BaseTypeUsed.size(); ++I)
    if (Run.BaseTypeUsed[I].load(std::memory_order_relaxed))
      Run.BaseUsedBytes += E.Pack->View.Types[I].length();
  for (size_t F = 0; F != Run.BaseFunctionUsed.size(); ++F)
    if (Run.BaseFunctionUsed[F].load(std::memory_order_relaxed))
      Run.BaseUsedBytes += E.Pack->View.Functions[F].size();
  uint64_t BaseBytes = Run.BaseBytes, BaseUsedBytes = Run.BaseUsedBytes;
  int64_t Missing = std::max<int64_t>(Run.MissingBytes.load(), 0);
  auto percent = [&](uint64_t N) {
    return BaseBytes ? double(N) * 100 / double(BaseBytes) : 0.0;
  };
  Message(Run.Options.Log) << "base " << idString(E.Id) << " (" << BaseBytes
                           << " record bytes): " << BaseUsedBytes << " used ("
                           << format("%.1f", percent(BaseUsedBytes)) << "%), "
                           << Missing << " more bytes a rebase would pack ("
                           << format("%.1f", percent(Missing)) << "%)";
  if (!Options.AutoRebase || Options.Base)
    return {};
  if (uint64_t(Missing) * 100 > BaseBytes * uint64_t(Options.MaxMissingPercent))
    return (Twine(Missing) +
            " bytes that a rebase would pack are not in "
            "the base, over " +
            Twine(Options.MaxMissingPercent) + "% of it")
        .str();
  if (BaseUsedBytes * 100 < BaseBytes * uint64_t(Options.MinUsedPercent))
    return (Twine("the inputs use less than ") + Twine(Options.MinUsedPercent) +
            "% of the base")
        .str();
  return {};
}

/// Against the base, each selected record's pack index is the index at which
/// the base holds its key.
static void assignBaseIndices(ProducerRun &Run) {
  for (InputObject &Object : Run.Objects) {
    if (!Object.packable()) {
      Object.Mapped = {};
      continue;
    }
    for (uint32_t &M : Object.Mapped)
      M = (M & SelectedBit)
              ? TypeIndex::FirstNonSimpleIndex + Run.baseIndex(M & ~SelectedBit)
              : 0;
    for (uint32_t S = 0; S != Object.SymbolMapped.size(); ++S)
      if (Object.SymbolMapped[S] == Selected)
        Object.SymbolMapped[S] =
            Run.BaseSymbols.lookup(Object.Symbols[S].KeyId);
    for (uint32_t F = 0; F != Object.FunctionMapped.size(); ++F)
      if (Object.FunctionMapped[F] == Selected)
        Object.FunctionMapped[F] =
            Run.BaseFunctions.lookup(Object.Functions[F].KeyId);
  }
}

/// A rebase assigns the new pack's indices in object and record order, so
/// that the pack is the same whatever the thread timing: each key gets one
/// index.
static void assignNewPackIndices(ProducerRun &Run) {
  std::vector<uint32_t> PackIndex(Run.KeyCount, 0);
  std::vector<uint32_t> SymbolOrdinal(Run.SymbolKeyCount, NoIndex);
  std::vector<uint32_t> FunctionOrdinal(Run.FunctionKeyCount, NoIndex);
  for (uint32_t O = 0; O != Run.Objects.size(); ++O) {
    InputObject &Object = Run.Objects[O];
    if (!Object.packable()) {
      Object.Mapped = {};
      continue;
    }
    bool AnyMapped = false;
    for (uint32_t I = 0; I != Object.Mapped.size(); ++I) {
      if (!(Object.Mapped[I] & SelectedBit)) {
        Object.Mapped[I] = 0;
        continue;
      }
      uint32_t Key = Object.Mapped[I] & ~SelectedBit;
      if (PackIndex[Key] == 0) {
        Run.Entries.push_back({O, I});
        PackIndex[Key] =
            TypeIndex::FirstNonSimpleIndex + Run.Entries.size() - 1;
      }
      Object.Mapped[I] = PackIndex[Key];
      AnyMapped = true;
    }
    Run.NeedNewPack |= AnyMapped;
    for (uint32_t S = 0; S != Object.SymbolMapped.size(); ++S) {
      if (Object.SymbolMapped[S] != Selected)
        continue;
      uint32_t Key = Object.Symbols[S].KeyId;
      if (SymbolOrdinal[Key] == NoIndex) {
        SymbolOrdinal[Key] = Run.SymbolEntries.size();
        Run.SymbolEntries.push_back({O, S});
      }
      Object.SymbolMapped[S] = SymbolOrdinal[Key];
    }
    // Functions, with the files they use.
    if (!AnyMapped) {
      Object.FunctionMapped = {};
      continue;
    }
    for (uint32_t F = 0; F != Object.FunctionMapped.size(); ++F) {
      if (Object.FunctionMapped[F] != Selected)
        continue;
      const FunctionDebug &Function = Object.Functions[F];
      uint32_t &Ordinal = FunctionOrdinal[Function.KeyId];
      if (Ordinal == NoIndex) {
        Ordinal = Run.FunctionEntries.size();
        Run.FunctionEntries.push_back({O, F});
        for (uint32_t File : ArrayRef<uint32_t>(Object.FunctionFiles)
                                 .slice(Function.FirstFile, Function.NumFiles))
          if (Run.FileOrdinal[File] == NoIndex) {
            Run.FileOrdinal[File] = Run.FileEntries.size();
            Run.FileEntries.push_back({O, File});
          }
      }
      Object.FunctionMapped[F] = Ordinal;
    }
  }
}

/// Phase 3b: give each selected record its pack index.
static void assignIndices(ProducerRun &Run) {
  Run.FileOrdinal.assign(Run.FileKeys->size(), NoIndex);
  if (Run.BaseMode)
    assignBaseIndices(Run);
  else
    assignNewPackIndices(Run);
}

/// Copy the records of the new pack that an object owns, renumbered into
/// the pack.
static Error copyOwnedRecords(ProducerRun &Run, size_t O,
                              NewPackRecords &Records) {
  const InputObject &Owner = Run.Objects[O];
  ObjectView View;
  if (Error E = openView(Owner, View))
    return E;
  // Every reference of a packed record is packed, so the owner's mapping
  // renumbers it into the new pack.
  for (uint32_t E : Records.TypesOwnedBy[O])
    Records.Types[E] =
        remapRecord(View.Types[Run.Entries[E].Record], false, Owner.Mapped);
  for (uint32_t S : Records.SymbolsOwnedBy[O]) {
    const SymbolCandidate &Symbol =
        Owner.Symbols[Run.SymbolEntries[S].Candidate];
    const SymbolSubsection &Subsection = Owner.Subsections[Symbol.Subsection];
    const SymbolRecordRef &Record = Subsection.Records[Symbol.Record];
    Records.Symbols[S] = remapRecord(
        View.section(Subsection.Section).slice(Record.Offset, Record.Size),
        true, Owner.Mapped);
  }
  if (Records.FunctionsOwnedBy[O].empty() && Records.FilesOwnedBy[O].empty())
    return Error::success();
  auto Files = ObjectFileTable::create(View.debugSSections(false));
  if (!Files)
    return Files.takeError();
  // The owner's files by key id, and the pack file of each offset.
  DenseMap<uint32_t, PackFileEntry> FileByKey;
  DenseMap<uint32_t, uint32_t> PackFileAt;
  for (const auto &[Offset, Entry] : Files->entries()) {
    if (Files->lookup(Entry) != Offset)
      continue;
    uint32_t Key = internFile(*Run.FileKeys, Entry);
    FileByKey.try_emplace(Key, Entry);
    if (Key < Run.FileOrdinal.size() && Run.FileOrdinal[Key] != NoIndex)
      PackFileAt.try_emplace(Offset, Run.FileOrdinal[Key]);
  }
  for (uint32_t F : Records.FilesOwnedBy[O]) {
    auto It = FileByKey.find(Run.FileEntries[F].KeyId);
    if (It == FileByKey.end())
      return producerError("a packed file is missing from its owner");
    Records.Files[F] = {It->second.Name.str(), It->second.ChecksumKind,
                        std::vector<uint8_t>(It->second.Checksum.begin(),
                                             It->second.Checksum.end())};
  }
  for (uint32_t F : Records.FunctionsOwnedBy[O]) {
    const FunctionDebug &Function =
        Owner.Functions[Run.FunctionEntries[F].Function];
    ArrayRef<uint8_t> Data = View.section(Function.Section);
    std::vector<uint8_t> Blob;
    Blob.reserve(Data.size());
    if (Error E = transformFunctionDebug(
            Data.drop_front(sizeof(uint32_t)), Blob,
            [&](uint32_t Index, uint32_t) -> Expected<uint32_t> {
              uint32_t I = Index - TypeIndex::FirstNonSimpleIndex;
              if (I >= Owner.Mapped.size() || !Owner.Mapped[I])
                return producerError("function references an inline type");
              return Owner.Mapped[I];
            },
            [&](uint32_t Checksum, uint32_t) -> Expected<uint32_t> {
              auto It = PackFileAt.find(Checksum);
              if (It == PackFileAt.end())
                return producerError("function references an unpacked file");
              return It->second;
            },
            FileOperandEncoding::Fixed))
      return E;
    Records.Functions[F] = std::move(Blob);
  }
  return Error::success();
}

/// Phase 4: write the new pack, and its index so that objects can be packed
/// against it one at a time. The records it holds are copied from their
/// owner objects, which are reopened in parallel.
static Error writeNewPack(ProducerRun &Run) {
  if (!Run.NeedNewPack)
    return Error::success();
  NewPackRecords Records(Run);
  parallelFor(0, Run.Objects.size(), [&](size_t O) {
    if (Records.TypesOwnedBy[O].empty() && Records.SymbolsOwnedBy[O].empty() &&
        Records.FunctionsOwnedBy[O].empty() && Records.FilesOwnedBy[O].empty())
      return;
    const InputObject &Owner = Run.Objects[O];
    Run.Budget.acquire(Owner.Size);
    scope_exit Release([&] { Run.Budget.release(Owner.Size); });
    if (Error E = copyOwnedRecords(Run, O, Records))
      Run.Errors[O] = toString(std::move(E));
  });
  for (size_t O = 0; O != Run.Objects.size(); ++O)
    if (!Run.Errors[O].empty())
      return producerError(Run.Objects[O].Path + ": " + Run.Errors[O]);

  PackContents Contents;
  Contents.Machine = Run.Machine;
  Contents.WriteGlobalHashes = true;
  for (const auto &Record : Records.Types)
    Contents.Types.push_back(Record);
  for (const auto &Record : Records.Symbols)
    Contents.Symbols.push_back(Record);
  for (const OwnedFile &File : Records.Files)
    Contents.Files.push_back({File.Name, File.ChecksumKind, File.Checksum});
  for (const auto &Function : Records.Functions)
    Contents.Functions.push_back(Function);
  auto Written = writePackFile(Contents);
  if (!Written)
    return Written.takeError();
  Run.NewPackId = Written->ContentId;
  Run.PackBytes = Written->FileData.size();
  SmallString<256> PackPath(Run.OutputPackDir);
  sys::path::append(PackPath,
                    idString(*Run.NewPackId) + PackFileExtension.str());
  if (Error E = writeFile(PackPath, Written->FileData))
    return E;
  auto Loaded = Run.Resolver.get(*Run.NewPackId);
  if (!Loaded)
    return Loaded.takeError();
  auto Index = buildPackIndex(**Loaded, *Run.NewPackId);
  if (!Index)
    return Index.takeError();
  auto Encoded = writePackIndex(std::move(*Index));
  if (!Encoded)
    return Encoded.takeError();
  SmallString<256> IndexPath(Run.OutputPackDir);
  sys::path::append(IndexPath,
                    idString(*Run.NewPackId) + PackIndexFileExtension.str());
  return writeFile(IndexPath, *Encoded);
}

/// The name of each staged object: that of its input, prefixed with a hash
/// of the input path when another input has the same name.
static Error nameStagedObjects(ProducerRun &Run) {
  Run.StagedNames.resize(Run.Objects.size());
  std::set<std::string> OutputNames;
  for (size_t O = 0; O != Run.Objects.size(); ++O) {
    const std::string &Path = Run.Objects[O].Path;
    std::string Name = sys::path::filename(Path).str();
    if (!OutputNames.insert(StringRef(Name).lower()).second) {
      Name = idString(BLAKE3::hash(arrayRefFromStringRef(Path))) + "-" + Name;
      if (!OutputNames.insert(StringRef(Name).lower()).second)
        return producerError("unable to create a unique staged object name");
    }
    Run.StagedNames[O] = Name;
  }
  return Error::success();
}

/// Stage an object: packed against the base or the new pack, or unchanged.
static void stageOneObject(ProducerRun &Run, size_t O) {
  InputObject &Object = Run.Objects[O];
  Run.Budget.acquire(Object.Size);
  scope_exit Release([&] { Run.Budget.release(Object.Size); });
  SmallString<256> OutputPath(Run.StagingDirectory);
  sys::path::append(OutputPath, Run.StagedNames[O]);
  std::vector<uint8_t> Data;
  std::optional<PackId> Target = Run.BaseMode ? Run.Base->Id : Run.NewPackId;
  if (Object.packable() && Target) {
    ObjectView View;
    Error E = openView(Object, View);
    Expected<StageResult> Staged =
        E ? Expected<StageResult>(std::move(E))
          : stageObject(Object, View, *Target, Run.Resolver);
    if (Staged) {
      Data = std::move(Staged->Data);
      Run.PackedFlags[O] = Staged->Packed;
      Run.MapEntries += Staged->MapEntries;
      Run.MapBytes += Staged->MapBytes;
      Run.StagedFunctions += Staged->Functions;
      Run.StagedFunctionBytes += Staged->FunctionBytes;
    } else {
      Run.Errors[O] = toString(Staged.takeError());
      ++Run.Fallbacks;
    }
  }
  if (Data.empty() && !Object.Materialized.empty()) {
    // A packed input that is not packed again: its original bytes, since
    // pruning may delete the pack it was packed against.
    Data = std::move(Object.Materialized);
  } else if (Data.empty()) {
    // Unchanged: a pass-through, unpacked or failed input, copied as is.
    if (Object.PassThrough)
      Run.PackedFlags[O] = 1;
    if (auto Buffer = MemoryBuffer::getFile(Object.Path, false, false))
      Data.assign((*Buffer)->getBufferStart(), (*Buffer)->getBufferEnd());
    else
      Run.Errors[O] = Buffer.getError().message();
  }
  Run.InputBytes += Object.Size;
  Run.OutputBytes += Data.size();
  if (Error E = writeFile(OutputPath, Data))
    Run.Errors[O] = toString(std::move(E));
  Object.Mapped = {};
  Object.Materialized = {};
}

/// Phase 5: stage every object in parallel.
static Error stageObjects(ProducerRun &Run) {
  if (Error E = nameStagedObjects(Run))
    return E;
  Run.PackedFlags.assign(Run.Objects.size(), 0);
  parallelFor(0, Run.Objects.size(), [&](size_t O) { stageOneObject(Run, O); });
  for (size_t O = 0; O != Run.Objects.size(); ++O) {
    const InputObject &Object = Run.Objects[O];
    if (!Run.Errors[O].empty())
      Message(Run.Options.Warn)
          << Object.Path << ": staged unchanged: " << Run.Errors[O];
    else if (Object.Unrestored)
      Message(Run.Options.Warn)
          << Object.Path << ": staged still packed: " << Object.SkipReason;
    else if (!Object.SkipReason.empty())
      Message(Run.Options.Log)
          << Object.Path << ": not packed: " << Object.SkipReason;
  }
  return Error::success();
}

/// When progress is logged, why the function debug info that stays in objects
/// is not packed.
static void logFunctionSkips(const ProducerRun &Run) {
  if (!Run.Options.Log)
    return;
  std::map<std::string, std::pair<uint64_t, uint64_t>> Skips;
  for (const InputObject &Object : Run.Objects)
    for (const auto &[Reason, Entry] : Object.FunctionSkips) {
      Skips[Reason].first += Entry.first;
      Skips[Reason].second += Entry.second;
    }
  for (const auto &[Reason, Entry] : Skips)
    Message(Run.Options.Log)
        << "function debug kept: " << Entry.first << " sections, "
        << Entry.second << " bytes: " << Reason;
}

/// Write cvpacks.json, which says what each staged object needs. A packed
/// object needs one pack: the base, or the new pack. A cache stores and
/// restores it with the object.
static Error writeManifest(const ProducerRun &Run, RunOutcome &Outcome) {
  std::vector<std::string> PackOfObjects;
  if (Run.BaseMode)
    PackOfObjects.push_back(idString(Run.Base->Id));
  else if (Run.NewPackId)
    PackOfObjects.push_back(idString(*Run.NewPackId));
  auto requiredPacks = [&](size_t O) -> ArrayRef<std::string> {
    if (!Run.PackedFlags[O])
      return {};
    return PackOfObjects;
  };
  std::vector<std::string> AllRequired;
  bool NewPackUsed = false;
  std::string Manifest;
  raw_string_ostream ManifestOS(Manifest);
  json::OStream J(ManifestOS, /*IndentSize=*/2);
  auto idList = [&](StringRef Key, ArrayRef<std::string> Ids) {
    J.attributeArray(Key, [&] {
      for (const std::string &Id : Ids)
        J.value(Id);
    });
  };
  J.object([&] {
    J.attributeArray("objects", [&] {
      for (size_t O = 0; O != Run.Objects.size(); ++O) {
        ArrayRef<std::string> Packs = requiredPacks(O);
        NewPackUsed |= Run.PackedFlags[O] && !Run.BaseMode;
        for (const std::string &Id : Packs)
          if (!llvm::is_contained(AllRequired, Id))
            AllRequired.push_back(Id);
        J.object([&] {
          J.attribute("source", Run.Objects[O].Path);
          J.attribute("staged", Run.StagedNames[O]);
          idList("packs", Packs);
        });
      }
    });
    idList("packs", AllRequired);
    idList("new_packs", NewPackUsed ? ArrayRef<std::string>(PackOfObjects)
                                    : ArrayRef<std::string>());
    if (Run.BaseMode) {
      J.attribute("base", idString(Run.Base->Id));
      J.attribute("base_bytes", Run.BaseBytes);
      J.attribute("base_used_bytes", Run.BaseUsedBytes);
      J.attribute("missing_bytes",
                  std::max<int64_t>(Run.MissingBytes.load(), 0));
    } else if (Run.NewPackId) {
      J.attribute("base", idString(*Run.NewPackId));
    }
    if (!Run.RebasedBecause.empty())
      J.attribute("rebased_because", Run.RebasedBecause);
  });
  ManifestOS << '\n';
  Outcome.Required = AllRequired;
  SmallString<256> ManifestPath(Run.StagingDirectory);
  sys::path::append(ManifestPath, "cvpacks.json");
  return writeFile(ManifestPath, arrayRefFromStringRef(Manifest));
}

static void logSummary(const ProducerRun &Run) {
  size_t Packed = llvm::count(Run.PackedFlags, 1);
  size_t Reusing = Run.BaseMode ? Packed : 0;
  Message(Run.Options.Log)
      << Run.Objects.size() << " objects, " << Packed << " packed (" << Reusing
      << " reusing existing packs), " << Run.Fallbacks
      << " staged unchanged after errors; " << Run.InputBytes << " -> "
      << Run.OutputBytes << " object bytes; new pack "
      << (Run.NewPackId ? idString(*Run.NewPackId) : std::string("<none>"))
      << " (" << Run.PackBytes << " bytes, " << Run.Entries.size()
      << " records, " << Run.SymbolEntries.size() << " symbols, "
      << Run.DedupSetSize << " packs in dedup set); " << Run.MapEntries
      << " type-map entries in " << Run.MapBytes << " bytes; "
      << Run.StagedFunctions << " function debug sections ("
      << Run.StagedFunctionBytes << " bytes) moved to "
      << Run.FunctionEntries.size() << " pack functions";
}

static Error runProducer(const PackOptions &Options,
                         ArrayRef<StringRef> InputPaths, bool Rebase,
                         StringRef RebasedBecause, RunOutcome &Outcome) {
  if (InputPaths.empty())
    return producerError("no input object to pack");
  if (Options.PackPaths.empty())
    return producerError("no pack directory");
  if (Options.StagingDirectory.empty())
    return producerError("no staging directory");
  if (std::error_code EC =
          sys::fs::create_directories(Options.StagingDirectory))
    return errorCodeToError(EC);
  if (std::error_code EC =
          sys::fs::create_directories(Options.PackPaths.front()))
    return errorCodeToError(EC);
  SmallVector<StringRef, 4> PackDirs;
  for (const std::string &Dir : Options.PackPaths)
    PackDirs.push_back(Dir);

  ProducerRun Run(Options, Rebase, RebasedBecause, PackDirs);
  Run.phase("start");
  analyzeInputs(Run, InputPaths);
  Run.phase("read and keyed");
  chooseMachine(Run);
  if (Error E = loadDedupSet(Run))
    return E;
  if (Error E = finishKeys(Run))
    return E;
  Run.phase("dedup set loaded");
  countKeys(Run);
  Run.phase("counted");
  chooseBase(Run);
  indexBase(Run);
  selectRecords(Run);
  std::string RebaseReason = checkBaseFit(Run);
  if (!RebaseReason.empty()) {
    Outcome.RebaseReason = std::move(RebaseReason);
    return Error::success();
  }
  Run.phase("selected");
  assignIndices(Run);
  Run.phase("assigned");
  if (Error E = writeNewPack(Run))
    return E;
  Run.phase("pack written");
  if (Error E = stageObjects(Run))
    return E;
  Run.phase("staged");
  logFunctionSkips(Run);
  if (Error E = writeManifest(Run, Outcome))
    return E;
  logSummary(Run);
  return Error::success();
}

/// Delete the packs of the output directory that no staged object needs.
static void prunePacks(const PackOptions &Options,
                       ArrayRef<std::string> RequiredPacks) {
  std::set<std::string> Required(RequiredPacks.begin(), RequiredPacks.end());
  std::error_code EC;
  uint64_t Count = 0, Bytes = 0;
  for (sys::fs::directory_iterator It(Options.PackPaths.front(), EC), End;
       It != End && !EC; It.increment(EC)) {
    StringRef Name = sys::path::filename(It->path());
    bool IsIndex = Name.ends_with_insensitive(PackIndexFileExtension);
    if (!IsIndex && !Name.ends_with_insensitive(PackFileExtension))
      continue;
    auto Id = parseIdString(Name.drop_back(
        IsIndex ? PackIndexFileExtension.size() : PackFileExtension.size()));
    if (!Id || Required.count(idString(*Id)))
      continue;
    uint64_t Size = 0;
    sys::fs::file_size(It->path(), Size);
    if (std::error_code RemoveEC = sys::fs::remove(It->path())) {
      Message(Options.Warn)
          << "cannot delete " << It->path() << ": " << RemoveEC.message();
      continue;
    }
    if (!IsIndex)
      ++Count;
    Bytes += Size;
  }
  Message(Options.Log) << "pruned " << Count << " packs (" << Bytes
                       << " bytes)";
}

Error llvm::codeview::cvpack::packObjects(const PackOptions &Options,
                                          ArrayRef<StringRef> InputPaths) {
  RunOutcome Outcome;
  std::string Reason;
  if (Error E =
          runProducer(Options, InputPaths, Options.Rebase, Reason, Outcome))
    return E;
  if (!Outcome.RebaseReason.empty()) {
    // Nothing was written: start over without the dedup set.
    Reason = Outcome.RebaseReason;
    Message(Options.Log) << "rebasing: " << Reason;
    Outcome = RunOutcome();
    if (Error E =
            runProducer(Options, InputPaths, /*Rebase=*/true, Reason, Outcome))
      return E;
  }
  // The runs' resolvers, which mapped the packs, are gone: a mapped file
  // cannot be deleted on Windows.
  if (Options.Prune)
    prunePacks(Options, Outcome.Required);
  return Error::success();
}
