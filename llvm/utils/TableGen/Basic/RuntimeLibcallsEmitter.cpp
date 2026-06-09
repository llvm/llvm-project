//===- RuntimeLibcallEmitter.cpp - Properties from RuntimeLibcalls.td -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "runtime-libcall-emitter"

#include "RuntimeLibcalls.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/TableGen/CodeGenHelpers.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/SetTheory.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <limits>

using namespace llvm;

namespace {
// Pair of a RuntimeLibcallPredicate and LibcallCallingConv to use as a map key.
struct PredicateWithCC {
  const Record *Predicate = nullptr;
  const Record *CallingConv = nullptr;

  PredicateWithCC() = default;
  PredicateWithCC(std::pair<const Record *, const Record *> P)
      : Predicate(P.first), CallingConv(P.second) {}

  PredicateWithCC(const Record *P, const Record *C)
      : Predicate(P), CallingConv(C) {}
};

inline bool operator==(PredicateWithCC LHS, PredicateWithCC RHS) {
  return LHS.Predicate == RHS.Predicate && LHS.CallingConv == RHS.CallingConv;
}
} // namespace

namespace llvm {
template <> struct DenseMapInfo<PredicateWithCC, void> {
  static unsigned getHashValue(const PredicateWithCC Val) {
    auto Pair = std::make_pair(Val.Predicate, Val.CallingConv);
    return DenseMapInfo<
        std::pair<const Record *, const Record *>>::getHashValue(Pair);
  }

  static bool isEqual(PredicateWithCC LHS, PredicateWithCC RHS) {
    return LHS == RHS;
  }
};

class RuntimeLibcallEmitter {
private:
  const RecordKeeper &Records;
  RuntimeLibcalls Libcalls;

  void emitGetRuntimeLibcallEnum(raw_ostream &OS) const;

  void emitNameMatchHashTable(raw_ostream &OS,
                              StringToOffsetTable &OffsetTable) const;

  void emitGetInitRuntimeLibcallNames(raw_ostream &OS) const;

  void emitSystemRuntimeLibrarySetCalls(raw_ostream &OS) const;

public:
  RuntimeLibcallEmitter(const RecordKeeper &R) : Records(R), Libcalls(R) {}

  void run(raw_ostream &OS);
};

} // End anonymous namespace.

void RuntimeLibcallEmitter::emitGetRuntimeLibcallEnum(raw_ostream &OS) const {
  IfDefEmitter IfDef(OS, "GET_RUNTIME_LIBCALL_ENUM");

  OS << "namespace llvm {\n"
        "namespace RTLIB {\n"
        "enum Libcall : unsigned short {\n";

  for (const RuntimeLibcall &LibCall : Libcalls.getRuntimeLibcallDefList()) {
    StringRef Name = LibCall.getName();
    OS << "  " << Name << " = " << LibCall.getEnumVal() << ",\n";
  }

  OS << "  UNKNOWN_LIBCALL = " << Libcalls.getRuntimeLibcallDefList().size()
     << "\n};\n\n"
        "enum LibcallImpl : unsigned short {\n"
        "  Unsupported = 0,\n";

  for (const RuntimeLibcallImpl &LibCall :
       Libcalls.getRuntimeLibcallImplDefList()) {
    OS << "  impl_" << LibCall.getName() << " = " << LibCall.getEnumVal()
       << ", // " << LibCall.getLibcallFuncName() << '\n';
  }

  OS << "};\n"
     << "constexpr size_t NumLibcallImpls = "
     << Libcalls.getRuntimeLibcallImplDefList().size() + 1
     << ";\n"
        "} // End namespace RTLIB\n"
        "} // End namespace llvm\n";
}

// StringMap uses xxh3_64bits, truncated to uint32_t.
static uint64_t hash(StringRef Str) {
  return static_cast<uint32_t>(xxh3_64bits(Str));
}

static void emitHashFunction(raw_ostream &OS) {
  OS << "static inline uint64_t hash(StringRef Str) {\n"
        "  return static_cast<uint32_t>(xxh3_64bits(Str));\n"
        "}\n\n";
}

/// Return the table size, maximum number of collisions for the set of hashes
static std::pair<int, int>
computePerfectHashParameters(ArrayRef<uint64_t> Hashes) {
  // Chosen based on experimentation with llvm/benchmarks/RuntimeLibcalls.cpp
  const int SizeOverhead = 4;

  // Index derived from hash -> number of collisions.
  DenseMap<uint64_t, int> Table;

  unsigned NumHashes = Hashes.size();

  for (int MaxCollisions = 1;; ++MaxCollisions) {
    for (unsigned N = NextPowerOf2(NumHashes - 1); N < SizeOverhead * NumHashes;
         N <<= 1) {
      Table.clear();

      bool NeedResize = false;
      for (uint64_t H : Hashes) {
        uint64_t Idx = H % static_cast<uint64_t>(N);
        if (++Table[Idx] > MaxCollisions) {
          // Need to resize the final table if we increased the collision count.
          NeedResize = true;
          break;
        }
      }

      if (!NeedResize)
        return {N, MaxCollisions};
    }
  }
}

static std::vector<unsigned>
constructPerfectHashTable(ArrayRef<RuntimeLibcallImpl> Keywords,
                          ArrayRef<uint64_t> Hashes,
                          ArrayRef<unsigned> TableValues, int Size,
                          int Collisions, StringToOffsetTable &OffsetTable) {
  std::vector<unsigned> Lookup(Size * Collisions);

  for (auto [HashValue, TableValue] : zip(Hashes, TableValues)) {
    uint64_t Idx = (HashValue % static_cast<uint64_t>(Size)) *
                   static_cast<uint64_t>(Collisions);

    bool Found = false;
    for (int J = 0; J < Collisions; ++J) {
      unsigned &Entry = Lookup[Idx + J];
      if (Entry == 0) {
        Entry = TableValue;
        Found = true;
        break;
      }
    }

    if (!Found)
      reportFatalInternalError("failure to hash");
  }

  return Lookup;
}

struct CompactHashTable {
  struct CollisionBucket {
    uint16_t Bucket;
    uint16_t Start;
    uint16_t Count;
  };

  std::vector<uint16_t> FirstValues;
  std::vector<uint16_t> SecondValues;
  std::vector<CollisionBucket> CollisionBuckets;
  std::vector<uint16_t> CollisionValues;
};

static CompactHashTable compactHashTable(ArrayRef<unsigned> Lookup,
                                         unsigned Collisions) {
  if (Collisions == 0 || Lookup.size() % Collisions != 0)
    reportFatalInternalError("invalid runtime libcall hash table");

  CompactHashTable Compact;
  const size_t NumBuckets = Lookup.size() / Collisions;
  Compact.FirstValues.reserve(NumBuckets);
  Compact.SecondValues.reserve(NumBuckets);

  for (size_t Bucket = 0; Bucket != NumBuckets; ++Bucket) {
    ArrayRef<unsigned> Values = Lookup.slice(Bucket * Collisions, Collisions);
    bool SawEmpty = false;
    for (unsigned Value : Values) {
      if (Value == 0) {
        SawEmpty = true;
        continue;
      }
      if (SawEmpty)
        reportFatalInternalError(
            "runtime libcall collisions are not contiguous");
      if (Value > std::numeric_limits<uint16_t>::max())
        reportFatalInternalError("runtime libcall index exceeds uint16_t");
    }

    Compact.FirstValues.push_back(Values[0]);
    Compact.SecondValues.push_back(Collisions > 1 ? Values[1] : 0);

    const size_t CollisionStart = Compact.CollisionValues.size();
    for (unsigned Value :
         Values.drop_front(std::min<unsigned>(Collisions, 2))) {
      if (Value == 0)
        break;
      Compact.CollisionValues.push_back(Value);
    }

    const size_t NumCollisions =
        Compact.CollisionValues.size() - CollisionStart;
    if (NumCollisions == 0)
      continue;
    if (Bucket > std::numeric_limits<uint16_t>::max() ||
        CollisionStart > std::numeric_limits<uint16_t>::max() ||
        NumCollisions > std::numeric_limits<uint16_t>::max())
      reportFatalInternalError(
          "runtime libcall collision metadata exceeds uint16_t");
    Compact.CollisionBuckets.push_back({static_cast<uint16_t>(Bucket),
                                        static_cast<uint16_t>(CollisionStart),
                                        static_cast<uint16_t>(NumCollisions)});
  }

  if (Compact.CollisionValues.size() > std::numeric_limits<uint16_t>::max())
    reportFatalInternalError(
        "too many runtime libcall collisions for uint16_t");

  return Compact;
}

/// Generate hash table based lookup by name.
void RuntimeLibcallEmitter::emitNameMatchHashTable(
    raw_ostream &OS, StringToOffsetTable &OffsetTable) const {
  ArrayRef<RuntimeLibcallImpl> RuntimeLibcallImplDefList =
      Libcalls.getRuntimeLibcallImplDefList();
  std::vector<uint64_t> Hashes(RuntimeLibcallImplDefList.size());
  std::vector<unsigned> TableValues(RuntimeLibcallImplDefList.size());
  DenseSet<StringRef> SeenFuncNames;

  size_t MaxFuncNameSize = 0;
  size_t Index = 0;

  for (const RuntimeLibcallImpl &LibCallImpl : RuntimeLibcallImplDefList) {
    StringRef ImplName = LibCallImpl.getLibcallFuncName();
    if (SeenFuncNames.insert(ImplName).second) {
      MaxFuncNameSize = std::max(MaxFuncNameSize, ImplName.size());
      TableValues[Index] = LibCallImpl.getEnumVal();
      Hashes[Index++] = hash(ImplName);
    }
  }

  // Trim excess elements from non-unique entries.
  Hashes.resize(SeenFuncNames.size());
  TableValues.resize(SeenFuncNames.size());

  LLVM_DEBUG({
    for (const RuntimeLibcallImpl &LibCallImpl : RuntimeLibcallImplDefList) {
      StringRef ImplName = LibCallImpl.getLibcallFuncName();
      if (ImplName.size() == MaxFuncNameSize) {
        dbgs() << "Maximum runtime libcall name size: " << ImplName << '('
               << MaxFuncNameSize << ")\n";
      }
    }
  });

  // Early exiting on the symbol name provides a significant speedup in the miss
  // case on the set of symbols in a clang binary. Emit this as an inlinable
  // precondition in the header.
  //
  // The empty check is also used to get sensible behavior on anonymous
  // functions.
  //
  // TODO: It may make more sense to split the search by string size more. There
  // are a few outliers, most call names are small.
  {
    IfDefEmitter IfDef(OS, "GET_LOOKUP_LIBCALL_IMPL_NAME_BODY");

    OS << "  size_t Size = Name.size();\n"
          "  if (Size == 0 || Size > "
       << MaxFuncNameSize
       << ")\n"
          "    return enum_seq(RTLIB::Unsupported, RTLIB::Unsupported);\n"
          " return lookupLibcallImplNameImpl(Name);\n";
  }

  auto [Size, Collisions] = computePerfectHashParameters(Hashes);
  std::vector<unsigned> Lookup =
      constructPerfectHashTable(RuntimeLibcallImplDefList, Hashes, TableValues,
                                Size, Collisions, OffsetTable);
  CompactHashTable Compact = compactHashTable(Lookup, Collisions);

  LLVM_DEBUG(dbgs() << "Runtime libcall perfect hashing parameters: Size = "
                    << Size << ", maximum collisions = " << Collisions << '\n');

  IfDefEmitter IfDef(OS, "DEFINE_GET_LOOKUP_LIBCALL_IMPL_NAME");
  emitHashFunction(OS);

  OS << "iota_range<RTLIB::LibcallImpl> RTLIB::RuntimeLibcallsInfo::"
        "lookupLibcallImplNameImpl(StringRef Name) {\n";

  OS << "  static constexpr uint16_t HashTableFirstValues[] = {\n";

  for (uint16_t Value : Compact.FirstValues)
    OS << "    " << Value << ",\n";

  OS << "  };\n\n";

  OS << "  static constexpr uint16_t HashTableSecondValues[] = {\n";

  for (uint16_t Value : Compact.SecondValues)
    OS << "    " << Value << ",\n";

  OS << "  };\n\n";

  const size_t CollisionBucketStorageSize =
      std::max<size_t>(Compact.CollisionBuckets.size(), 1);
  const size_t CollisionValueStorageSize =
      std::max<size_t>(Compact.CollisionValues.size(), 1);
  OS << "  struct HashTableCollisionBucket {\n"
        "    uint16_t Bucket;\n"
        "    uint16_t Start;\n"
        "    uint16_t Count;\n"
        "  };\n\n"
        "  static constexpr HashTableCollisionBucket "
        "HashTableCollisionBuckets[] = {\n";

  for (const CompactHashTable::CollisionBucket &Bucket :
       Compact.CollisionBuckets)
    OS << "    {" << Bucket.Bucket << ", " << Bucket.Start << ", "
       << Bucket.Count << "},\n";
  if (Compact.CollisionBuckets.empty())
    OS << "    {0, 0, 0},\n";

  OS << "  };\n\n"
        "  static constexpr uint16_t HashTableCollisionValues[] = {\n";

  for (uint16_t Value : Compact.CollisionValues)
    OS << "    " << Value << ",\n";
  if (Compact.CollisionValues.empty())
    OS << "    0,\n";

  OS << "  };\n\n";

  OS << "  static_assert(RTLIB::NumLibcallImpls <= "
        "uint32_t(UINT16_MAX) + 1,\n"
        "                \"runtime libcall indices must fit in uint16_t\");\n"
     << "  static_assert(" << Size
     << " <= uint32_t(UINT16_MAX) + 1,\n"
        "                \"runtime libcall bucket indices must fit in "
        "uint16_t\");\n"
     << "  static_assert(" << Compact.CollisionValues.size()
     << " <= UINT16_MAX,\n"
        "                \"runtime libcall collision offsets must fit in "
        "uint16_t\");\n"
     << "  static_assert(sizeof(HashTableFirstValues) / "
        "sizeof(HashTableFirstValues[0]) == "
     << Compact.FirstValues.size() << ");\n"
     << "  static_assert(sizeof(HashTableSecondValues) / "
        "sizeof(HashTableSecondValues[0]) == "
     << Compact.SecondValues.size() << ");\n"
     << "  static_assert(sizeof(HashTableCollisionBucket) == 6);\n"
     << "  static_assert(sizeof(HashTableCollisionBuckets) / "
        "sizeof(HashTableCollisionBuckets[0]) == "
     << CollisionBucketStorageSize << ");\n"
     << "  static_assert(sizeof(HashTableCollisionValues) / "
        "sizeof(HashTableCollisionValues[0]) == "
     << CollisionValueStorageSize << ");\n\n";

  OS << "  unsigned Idx = hash(Name) % " << Size
     << ";\n"
        "  uint16_t Entry = HashTableFirstValues[Idx];\n"
        "  uint16_t StrOffset = RuntimeLibcallNameOffsetTable[Entry];\n"
        "  uint8_t StrSize = RuntimeLibcallNameSizeTable[Entry];\n"
        "  StringRef Str(\n"
        "    &RTLIB::RuntimeLibcallsInfo::"
        "RuntimeLibcallImplNameTableStorage[StrOffset],\n"
        "    StrSize);\n"
        "  if (Str != Name) {\n"
        "    Entry = HashTableSecondValues[Idx];\n"
        "    if (Entry == 0)\n"
        "      return enum_seq(RTLIB::Unsupported, "
        "RTLIB::Unsupported);\n\n"
        "    StrOffset = RuntimeLibcallNameOffsetTable[Entry];\n"
        "    StrSize = RuntimeLibcallNameSizeTable[Entry];\n"
        "    Str = StringRef(\n"
        "      &RTLIB::RuntimeLibcallsInfo::"
        "RuntimeLibcallImplNameTableStorage[StrOffset],\n"
        "      StrSize);\n"
        "    if (Str != Name) {\n"
        "      auto CollisionBuckets = ArrayRef(HashTableCollisionBuckets, "
     << Compact.CollisionBuckets.size()
     << ");\n"
        "      auto It = llvm::lower_bound(\n"
        "          CollisionBuckets, Idx,\n"
        "          [](const HashTableCollisionBucket &Bucket, "
        "unsigned Idx) {\n"
        "            return Bucket.Bucket < Idx;\n"
        "          });\n"
        "      if (It == CollisionBuckets.end() || It->Bucket != Idx)\n"
        "        return enum_seq(RTLIB::Unsupported, "
        "RTLIB::Unsupported);\n\n"
        "      for (unsigned I = It->Start, E = I + It->Count; I != E; "
        "++I) {\n"
        R"(        Entry = HashTableCollisionValues[I];
        StrOffset = RuntimeLibcallNameOffsetTable[Entry];
        StrSize = RuntimeLibcallNameSizeTable[Entry];
        Str = StringRef(
        &RTLIB::RuntimeLibcallsInfo::RuntimeLibcallImplNameTableStorage[StrOffset],
        StrSize);
        if (Str == Name)
          return libcallImplNameHit(Entry, StrOffset);
      }

      return enum_seq(RTLIB::Unsupported, RTLIB::Unsupported);
    }
  }

  return libcallImplNameHit(Entry, StrOffset);
}
)";
}

void RuntimeLibcallEmitter::emitGetInitRuntimeLibcallNames(
    raw_ostream &OS) const {
  // Emit the implementation names
  StringToOffsetTable Table(/*AppendZero=*/true,
                            "RTLIB::RuntimeLibcallsInfo::");

  {
    IfDefEmitter IfDef(OS, "GET_INIT_RUNTIME_LIBCALL_NAMES");

    for (const RuntimeLibcallImpl &LibCallImpl :
         Libcalls.getRuntimeLibcallImplDefList())
      Table.GetOrAddStringOffset(LibCallImpl.getLibcallFuncName());

    Table.EmitStringTableDef(OS, "RuntimeLibcallImplNameTable");
    OS << R"(
const uint16_t RTLIB::RuntimeLibcallsInfo::RuntimeLibcallNameOffsetTable[] = {
)";

    OS << formatv("  {}, // {}\n", Table.GetStringOffset(""),
                  ""); // Unsupported entry
    for (const RuntimeLibcallImpl &LibCallImpl :
         Libcalls.getRuntimeLibcallImplDefList()) {
      StringRef ImplName = LibCallImpl.getLibcallFuncName();
      OS << formatv("  {}, // {}\n", Table.GetStringOffset(ImplName), ImplName);
    }
    OS << "};\n";

    OS << R"(
const uint8_t RTLIB::RuntimeLibcallsInfo::RuntimeLibcallNameSizeTable[] = {
)";

    OS << "  0,\n";
    for (const RuntimeLibcallImpl &LibCallImpl :
         Libcalls.getRuntimeLibcallImplDefList())
      OS << "  " << LibCallImpl.getLibcallFuncName().size() << ",\n";
    OS << "};\n\n";

    // Emit the reverse mapping from implementation libraries to RTLIB::Libcall
    OS << "const RTLIB::Libcall llvm::RTLIB::RuntimeLibcallsInfo::"
          "ImplToLibcall[RTLIB::NumLibcallImpls] = {\n"
          "  RTLIB::UNKNOWN_LIBCALL, // RTLIB::Unsupported\n";

    for (const RuntimeLibcallImpl &LibCallImpl :
         Libcalls.getRuntimeLibcallImplDefList()) {
      const RuntimeLibcall *Provides = LibCallImpl.getProvides();
      OS << "  ";
      Provides->emitEnumEntry(OS);
      OS << ", // ";
      LibCallImpl.emitEnumEntry(OS);
      OS << '\n';
    }

    OS << "};\n\n";
  }

  emitNameMatchHashTable(OS, Table);
}

void RuntimeLibcallEmitter::emitSystemRuntimeLibrarySetCalls(
    raw_ostream &OS) const {
  OS << "void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets("
        "const llvm::Triple &TT, ExceptionHandling ExceptionModel, "
        "FloatABI::ABIType FloatABI, EABI EABIVersion, "
        "StringRef ABIName) {\n";

  ArrayRef<const Record *> AllLibs =
      Records.getAllDerivedDefinitions("SystemRuntimeLibrary");

  for (const Record *R : AllLibs) {
    OS << '\n';

    AvailabilityPredicate TopLevelPredicate(R->getValueAsDef("TriplePred"));

    OS << indent(2);
    TopLevelPredicate.emitIf(OS);

    if (const Record *DefaultCCClass =
            R->getValueAsDef("DefaultLibcallCallingConv")) {
      StringRef DefaultCC =
          DefaultCCClass->getValueAsString("CallingConv").trim();

      if (!DefaultCC.empty()) {
        OS << "    const CallingConv::ID DefaultCC = " << DefaultCC << ";\n"
           << "    for (CallingConv::ID &Entry : LibcallImplCallingConvs) {\n"
              "      Entry = DefaultCC;\n"
              "    }\n\n";
      }
    }

    SetTheory Sets;

    DenseMap<const RuntimeLibcallImpl *,
             std::pair<std::vector<const Record *>, const Record *>>
        Func2Preds;
    Sets.addExpander("LibcallImpls", std::make_unique<LibcallPredicateExpander>(
                                         Libcalls, Func2Preds));

    const SetTheory::RecVec *Elements =
        Sets.expand(R->getValueAsDef("MemberList"));

    // Sort to get deterministic output
    SetVector<PredicateWithCC> PredicateSorter;
    PredicateSorter.insert(
        PredicateWithCC()); // No predicate or CC override first.

    constexpr unsigned BitsPerStorageElt = 64;
    DenseMap<PredicateWithCC, LibcallsWithCC> Pred2Funcs;

    SmallVector<uint64_t, 32> BitsetValues(divideCeil(
        Libcalls.getRuntimeLibcallImplDefList().size() + 1, BitsPerStorageElt));

    for (const Record *Elt : *Elements) {
      const RuntimeLibcallImpl *LibCallImpl =
          Libcalls.getRuntimeLibcallImpl(Elt);
      if (!LibCallImpl) {
        PrintError(R, "entry for SystemLibrary is not a RuntimeLibcallImpl");
        PrintNote(Elt->getLoc(), "invalid entry `" + Elt->getName() + "`");
        continue;
      }

      size_t BitIdx = LibCallImpl->getEnumVal();
      uint64_t BitmaskVal = uint64_t(1) << (BitIdx % BitsPerStorageElt);
      size_t BitsetIdx = BitIdx / BitsPerStorageElt;

      auto It = Func2Preds.find(LibCallImpl);
      if (It == Func2Preds.end()) {
        BitsetValues[BitsetIdx] |= BitmaskVal;
        Pred2Funcs[PredicateWithCC()].LibcallImpls.push_back(LibCallImpl);
        continue;
      }

      for (const Record *Pred : It->second.first) {
        const Record *CC = It->second.second;
        AvailabilityPredicate SubsetPredicate(Pred);
        if (SubsetPredicate.isAlwaysAvailable())
          BitsetValues[BitsetIdx] |= BitmaskVal;

        PredicateWithCC Key(Pred, CC);
        auto &Entry = Pred2Funcs[Key];
        Entry.LibcallImpls.push_back(LibCallImpl);
        Entry.CallingConv = It->second.second;
        PredicateSorter.insert(Key);
      }
    }

    OS << "    static constexpr LibcallImplBitset SystemAvailableImpls({\n"
       << indent(6);

    ListSeparator LS;
    unsigned EntryCount = 0;
    for (uint64_t Bits : BitsetValues) {
      if (EntryCount++ == 4) {
        EntryCount = 1;
        OS << ",\n" << indent(6);
      } else
        OS << LS;
      OS << format_hex(Bits, 16);
    }
    OS << "\n    });\n"
          "    AvailableLibcallImpls = SystemAvailableImpls;\n\n";

    SmallVector<PredicateWithCC, 0> SortedPredicates =
        PredicateSorter.takeVector();

    llvm::sort(SortedPredicates, [](PredicateWithCC A, PredicateWithCC B) {
      StringRef AName = A.Predicate ? A.Predicate->getName() : "";
      StringRef BName = B.Predicate ? B.Predicate->getName() : "";
      return AName < BName;
    });

    for (PredicateWithCC Entry : SortedPredicates) {
      AvailabilityPredicate SubsetPredicate(Entry.Predicate);
      unsigned IndentDepth = 2;

      auto It = Pred2Funcs.find(Entry);
      if (It == Pred2Funcs.end())
        continue;

      if (!SubsetPredicate.isAlwaysAvailable()) {
        IndentDepth = 4;

        OS << indent(IndentDepth);
        SubsetPredicate.emitIf(OS);
      }

      LibcallsWithCC &FuncsWithCC = It->second;

      std::vector<const RuntimeLibcallImpl *> &Funcs = FuncsWithCC.LibcallImpls;

      // Ensure we only emit a unique implementation per libcall in the
      // selection table.
      //
      // FIXME: We need to generate separate functions for
      // is-libcall-available and should-libcall-be-used to avoid this.
      //
      // This also makes it annoying to make use of the default set, since the
      // entries from the default set may win over the replacements unless
      // they are explicitly removed.
      stable_sort(Funcs, [](const RuntimeLibcallImpl *A,
                            const RuntimeLibcallImpl *B) {
        return A->getProvides()->getEnumVal() < B->getProvides()->getEnumVal();
      });

      auto UniqueI = llvm::unique(
          Funcs, [&](const RuntimeLibcallImpl *A, const RuntimeLibcallImpl *B) {
            if (A->getProvides() == B->getProvides()) {
              PrintWarning(R->getLoc(),
                           Twine("conflicting implementations for libcall " +
                                 A->getProvides()->getName() + ": " +
                                 A->getLibcallFuncName() + ", " +
                                 B->getLibcallFuncName()));
              return true;
            }

            return false;
          });

      Funcs.erase(UniqueI, Funcs.end());

      OS << indent(IndentDepth + 2)
         << "static const RTLIB::LibcallImpl LibraryCalls";
      SubsetPredicate.emitTableVariableNameSuffix(OS);
      if (FuncsWithCC.CallingConv)
        OS << '_' << FuncsWithCC.CallingConv->getName();

      OS << "[] = {\n";
      for (const RuntimeLibcallImpl *LibCallImpl : Funcs) {
        OS << indent(IndentDepth + 6);
        LibCallImpl->emitEnumEntry(OS);
        OS << ", // " << LibCallImpl->getLibcallFuncName() << '\n';
      }

      OS << indent(IndentDepth + 2) << "};\n\n"
         << indent(IndentDepth + 2)
         << "for (const RTLIB::LibcallImpl Impl : LibraryCalls";
      SubsetPredicate.emitTableVariableNameSuffix(OS);
      if (FuncsWithCC.CallingConv)
        OS << '_' << FuncsWithCC.CallingConv->getName();

      OS << ") {\n" << indent(IndentDepth + 4) << "setAvailable(Impl);\n";

      if (FuncsWithCC.CallingConv) {
        StringRef CCEnum =
            FuncsWithCC.CallingConv->getValueAsString("CallingConv");
        OS << indent(IndentDepth + 4) << "setLibcallImplCallingConv(Impl, "
           << CCEnum << ");\n";
      }

      OS << indent(IndentDepth + 2) << "}\n";
      OS << '\n';

      if (!SubsetPredicate.isAlwaysAvailable()) {
        OS << indent(IndentDepth);
        SubsetPredicate.emitEndIf(OS);
        OS << '\n';
      }
    }

    OS << indent(4) << "return;\n" << indent(2);
    TopLevelPredicate.emitEndIf(OS);
  }

  // FIXME: This should be a fatal error. A few contexts are improperly relying
  // on RuntimeLibcalls constructed with fully unknown triples.
  OS << "  LLVM_DEBUG(dbgs() << \"no system runtime library applied to target "
        "\\'\" << TT.str() << \"\\'\\n\");\n"
        "}\n\n";
}

void RuntimeLibcallEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Runtime LibCalls Source Fragment", OS, Records);
  emitGetRuntimeLibcallEnum(OS);

  emitGetInitRuntimeLibcallNames(OS);

  {
    IfDefEmitter IfDef(OS, "GET_RUNTIME_LIBCALLS_INFO");
    emitSystemRuntimeLibrarySetCalls(OS);
  }
}

static TableGen::Emitter::OptClass<RuntimeLibcallEmitter>
    X("gen-runtime-libcalls", "Generate RuntimeLibcalls");
