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
  static inline PredicateWithCC getEmptyKey() {
    return DenseMapInfo<
        std::pair<const Record *, const Record *>>::getEmptyKey();
  }

  static inline PredicateWithCC getTombstoneKey() {
    return DenseMapInfo<
        std::pair<const Record *, const Record *>>::getTombstoneKey();
  }

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

  LLVM_DEBUG(dbgs() << "Runtime libcall perfect hashing parameters: Size = "
                    << Size << ", maximum collisions = " << Collisions << '\n');

  IfDefEmitter IfDef(OS, "DEFINE_GET_LOOKUP_LIBCALL_IMPL_NAME");
  emitHashFunction(OS);

  OS << "iota_range<RTLIB::LibcallImpl> RTLIB::RuntimeLibcallsInfo::"
        "lookupLibcallImplNameImpl(StringRef Name) {\n";

  // Emit RTLIB::LibcallImpl values
  OS << "  static constexpr uint16_t HashTableNameToEnum[" << Lookup.size()
     << "] = {\n";

  for (unsigned TableVal : Lookup)
    OS << "    " << TableVal << ",\n";

  OS << "  };\n\n";

  OS << "  unsigned Idx = (hash(Name) % " << Size << ") * " << Collisions
     << ";\n\n"
        "  for (int I = 0; I != "
     << Collisions << R"(; ++I) {
    const uint16_t Entry = HashTableNameToEnum[Idx + I];
    const uint16_t StrOffset = RuntimeLibcallNameOffsetTable[Entry];
    const uint8_t StrSize = RuntimeLibcallNameSizeTable[Entry];
    StringRef Str(
      &RTLIB::RuntimeLibcallsInfo::RuntimeLibcallImplNameTableStorage[StrOffset],
      StrSize);
    if (Str == Name)
      return libcallImplNameHit(Entry, StrOffset);
  }

  return enum_seq(RTLIB::Unsupported, RTLIB::Unsupported);
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
