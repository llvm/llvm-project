//===- RuntimeLibcallEmitter.cpp - Properties from RuntimeLibcalls.td -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "runtime-libcall-emitter"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
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
} // namespace llvm

namespace {

class AvailabilityPredicate {
  const Record *TheDef;
  StringRef PredicateString;

public:
  AvailabilityPredicate(const Record *Def) : TheDef(Def) {
    if (TheDef)
      PredicateString = TheDef->getValueAsString("Cond");
  }

  const Record *getDef() const { return TheDef; }

  bool isAlwaysAvailable() const { return PredicateString.empty(); }

  void emitIf(raw_ostream &OS) const {
    OS << "if (" << PredicateString << ") {\n";
  }

  void emitEndIf(raw_ostream &OS) const { OS << "}\n"; }

  void emitTableVariableNameSuffix(raw_ostream &OS) const {
    if (TheDef)
      OS << '_' << TheDef->getName();
  }
};

class RuntimeLibcallEmitter;
class RuntimeLibcallImpl;

/// Used to apply predicates to nested sets of libcalls.
struct LibcallPredicateExpander : SetTheory::Expander {
  const RuntimeLibcallEmitter &LibcallEmitter;
  DenseMap<const RuntimeLibcallImpl *,
           std::pair<std::vector<const Record *>, const Record *>> &Func2Preds;

  LibcallPredicateExpander(
      const RuntimeLibcallEmitter &LibcallEmitter,
      DenseMap<const RuntimeLibcallImpl *,
               std::pair<std::vector<const Record *>, const Record *>>
          &Func2Preds)
      : LibcallEmitter(LibcallEmitter), Func2Preds(Func2Preds) {}

  void expand(SetTheory &ST, const Record *Def,
              SetTheory::RecSet &Elts) override;
};

class RuntimeLibcall {
  const Record *TheDef = nullptr;
  const size_t EnumVal;

public:
  RuntimeLibcall() = delete;
  RuntimeLibcall(const Record *Def, size_t EnumVal)
      : TheDef(Def), EnumVal(EnumVal) {
    assert(Def);
  }

  ~RuntimeLibcall() { assert(TheDef); }

  const Record *getDef() const { return TheDef; }

  StringRef getName() const { return TheDef->getName(); }

  size_t getEnumVal() const { return EnumVal; }

  void emitEnumEntry(raw_ostream &OS) const {
    OS << "RTLIB::" << TheDef->getValueAsString("Name");
  }
};

class RuntimeLibcallImpl {
  const Record *TheDef;
  const RuntimeLibcall *Provides = nullptr;
  const size_t EnumVal;

public:
  RuntimeLibcallImpl(
      const Record *Def,
      const DenseMap<const Record *, const RuntimeLibcall *> &ProvideMap,
      size_t EnumVal)
      : TheDef(Def), EnumVal(EnumVal) {
    if (const Record *ProvidesDef = Def->getValueAsDef("Provides"))
      Provides = ProvideMap.lookup(ProvidesDef);
  }

  ~RuntimeLibcallImpl() {}

  const Record *getDef() const { return TheDef; }

  StringRef getName() const { return TheDef->getName(); }

  size_t getEnumVal() const { return EnumVal; }

  const RuntimeLibcall *getProvides() const { return Provides; }

  StringRef getLibcallFuncName() const {
    return TheDef->getValueAsString("LibCallFuncName");
  }

  const Record *getCallingConv() const {
    return TheDef->getValueAsOptionalDef("CallingConv");
  }

  void emitQuotedLibcallFuncName(raw_ostream &OS) const {
    OS << '\"' << getLibcallFuncName() << '\"';
  }

  bool isDefault() const { return TheDef->getValueAsBit("IsDefault"); }

  void emitEnumEntry(raw_ostream &OS) const {
    OS << "RTLIB::impl_" << this->getName();
  }

  void emitSetImplCall(raw_ostream &OS) const {
    OS << "setLibcallImpl(";
    Provides->emitEnumEntry(OS);
    OS << ", ";
    emitEnumEntry(OS);
    OS << "); // " << getLibcallFuncName() << '\n';
  }

  void emitTableEntry(raw_ostream &OS) const {
    OS << '{';
    Provides->emitEnumEntry(OS);
    OS << ", ";
    emitEnumEntry(OS);
    OS << "}, // " << getLibcallFuncName() << '\n';
  }

  void emitSetCallingConv(raw_ostream &OS) const {}
};

struct LibcallsWithCC {
  std::vector<const RuntimeLibcallImpl *> LibcallImpls;
  const Record *CallingConv = nullptr;
};

class RuntimeLibcallEmitter {
private:
  const RecordKeeper &Records;
  DenseMap<const Record *, const RuntimeLibcall *> Def2RuntimeLibcall;
  DenseMap<const Record *, const RuntimeLibcallImpl *> Def2RuntimeLibcallImpl;

  std::vector<RuntimeLibcall> RuntimeLibcallDefList;
  std::vector<RuntimeLibcallImpl> RuntimeLibcallImplDefList;

  DenseMap<const RuntimeLibcall *, const RuntimeLibcallImpl *>
      LibCallToDefaultImpl;

private:
  void emitGetRuntimeLibcallEnum(raw_ostream &OS) const;

  void emitNameMatchHashTable(raw_ostream &OS,
                              StringToOffsetTable &OffsetTable) const;

  void emitGetInitRuntimeLibcallNames(raw_ostream &OS) const;

  void emitSystemRuntimeLibrarySetCalls(raw_ostream &OS) const;

public:
  RuntimeLibcallEmitter(const RecordKeeper &R) : Records(R) {

    ArrayRef<const Record *> AllRuntimeLibcalls =
        Records.getAllDerivedDefinitions("RuntimeLibcall");

    RuntimeLibcallDefList.reserve(AllRuntimeLibcalls.size());

    size_t CallTypeEnumVal = 0;
    for (const Record *RuntimeLibcallDef : AllRuntimeLibcalls) {
      RuntimeLibcallDefList.emplace_back(RuntimeLibcallDef, CallTypeEnumVal++);
      Def2RuntimeLibcall[RuntimeLibcallDef] = &RuntimeLibcallDefList.back();
    }

    for (RuntimeLibcall &LibCall : RuntimeLibcallDefList)
      Def2RuntimeLibcall[LibCall.getDef()] = &LibCall;

    ArrayRef<const Record *> AllRuntimeLibcallImplsRaw =
        Records.getAllDerivedDefinitions("RuntimeLibcallImpl");

    SmallVector<const Record *, 1024> AllRuntimeLibcallImpls(
        AllRuntimeLibcallImplsRaw);

    // Sort by libcall impl name and secondarily by the enum name.
    sort(AllRuntimeLibcallImpls, [](const Record *A, const Record *B) {
      return std::pair(A->getValueAsString("LibCallFuncName"), A->getName()) <
             std::pair(B->getValueAsString("LibCallFuncName"), B->getName());
    });

    RuntimeLibcallImplDefList.reserve(AllRuntimeLibcallImpls.size());

    size_t LibCallImplEnumVal = 1;
    for (const Record *LibCallImplDef : AllRuntimeLibcallImpls) {
      RuntimeLibcallImplDefList.emplace_back(LibCallImplDef, Def2RuntimeLibcall,
                                             LibCallImplEnumVal++);

      const RuntimeLibcallImpl &LibCallImpl = RuntimeLibcallImplDefList.back();
      Def2RuntimeLibcallImpl[LibCallImplDef] = &LibCallImpl;

      if (LibCallImpl.isDefault()) {
        const RuntimeLibcall *Provides = LibCallImpl.getProvides();
        if (!Provides)
          PrintFatalError(LibCallImplDef->getLoc(),
                          "default implementations must provide a libcall");
        LibCallToDefaultImpl[Provides] = &LibCallImpl;
      }
    }
  }

  const RuntimeLibcall *getRuntimeLibcall(const Record *Def) const {
    return Def2RuntimeLibcall.lookup(Def);
  }

  const RuntimeLibcallImpl *getRuntimeLibcallImpl(const Record *Def) const {
    return Def2RuntimeLibcallImpl.lookup(Def);
  }

  void run(raw_ostream &OS);
};

} // End anonymous namespace.

void RuntimeLibcallEmitter::emitGetRuntimeLibcallEnum(raw_ostream &OS) const {
  OS << "#ifdef GET_RUNTIME_LIBCALL_ENUM\n"
        "namespace llvm {\n"
        "namespace RTLIB {\n"
        "enum Libcall : unsigned short {\n";

  for (const RuntimeLibcall &LibCall : RuntimeLibcallDefList) {
    StringRef Name = LibCall.getName();
    OS << "  " << Name << " = " << LibCall.getEnumVal() << ",\n";
  }

  OS << "  UNKNOWN_LIBCALL = " << RuntimeLibcallDefList.size()
     << "\n};\n\n"
        "enum LibcallImpl : unsigned short {\n"
        "  Unsupported = 0,\n";

  for (const RuntimeLibcallImpl &LibCall : RuntimeLibcallImplDefList) {
    OS << "  impl_" << LibCall.getName() << " = " << LibCall.getEnumVal()
       << ", // " << LibCall.getLibcallFuncName() << '\n';
  }

  OS << "};\n"
     << "constexpr size_t NumLibcallImpls = "
     << RuntimeLibcallImplDefList.size() + 1
     << ";\n"
        "} // End namespace RTLIB\n"
        "} // End namespace llvm\n"
        "#endif\n\n";
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
  OS << "#ifdef GET_LOOKUP_LIBCALL_IMPL_NAME_BODY\n"
        "  size_t Size = Name.size();\n"
        "  if (Size == 0 || Size > "
     << MaxFuncNameSize
     << ")\n"
        "    return enum_seq(RTLIB::Unsupported, RTLIB::Unsupported);\n"
        " return lookupLibcallImplNameImpl(Name);\n"
        "#endif\n";

  auto [Size, Collisions] = computePerfectHashParameters(Hashes);
  std::vector<unsigned> Lookup =
      constructPerfectHashTable(RuntimeLibcallImplDefList, Hashes, TableValues,
                                Size, Collisions, OffsetTable);

  LLVM_DEBUG(dbgs() << "Runtime libcall perfect hashing parameters: Size = "
                    << Size << ", maximum collisions = " << Collisions << '\n');

  OS << "#ifdef DEFINE_GET_LOOKUP_LIBCALL_IMPL_NAME\n";
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

  OS << "#endif\n\n";
}

void RuntimeLibcallEmitter::emitGetInitRuntimeLibcallNames(
    raw_ostream &OS) const {
  OS << "#ifdef GET_INIT_RUNTIME_LIBCALL_NAMES\n";

  // Emit the implementation names
  StringToOffsetTable Table(/*AppendZero=*/true,
                            "RTLIB::RuntimeLibcallsInfo::");

  for (const RuntimeLibcallImpl &LibCallImpl : RuntimeLibcallImplDefList)
    Table.GetOrAddStringOffset(LibCallImpl.getLibcallFuncName());

  Table.EmitStringTableDef(OS, "RuntimeLibcallImplNameTable");
  OS << R"(
const uint16_t RTLIB::RuntimeLibcallsInfo::RuntimeLibcallNameOffsetTable[] = {
)";

  OS << formatv("  {}, // {}\n", Table.GetStringOffset(""),
                ""); // Unsupported entry
  for (const RuntimeLibcallImpl &LibCallImpl : RuntimeLibcallImplDefList) {
    StringRef ImplName = LibCallImpl.getLibcallFuncName();
    OS << formatv("  {}, // {}\n", Table.GetStringOffset(ImplName), ImplName);
  }
  OS << "};\n";

  OS << R"(
const uint8_t RTLIB::RuntimeLibcallsInfo::RuntimeLibcallNameSizeTable[] = {
)";

  OS << "  0,\n";
  for (const RuntimeLibcallImpl &LibCallImpl : RuntimeLibcallImplDefList)
    OS << "  " << LibCallImpl.getLibcallFuncName().size() << ",\n";
  OS << "};\n\n";

  // Emit the reverse mapping from implementation libraries to RTLIB::Libcall
  OS << "const RTLIB::Libcall llvm::RTLIB::RuntimeLibcallsInfo::"
        "ImplToLibcall[RTLIB::NumLibcallImpls] = {\n"
        "  RTLIB::UNKNOWN_LIBCALL, // RTLIB::Unsupported\n";

  for (const RuntimeLibcallImpl &LibCallImpl : RuntimeLibcallImplDefList) {
    const RuntimeLibcall *Provides = LibCallImpl.getProvides();
    OS << "  ";
    Provides->emitEnumEntry(OS);
    OS << ", // ";
    LibCallImpl.emitEnumEntry(OS);
    OS << '\n';
  }
  OS << "};\n\n";

  OS << "#endif\n\n";

  emitNameMatchHashTable(OS, Table);
}

void RuntimeLibcallEmitter::emitSystemRuntimeLibrarySetCalls(
    raw_ostream &OS) const {
  OS << "void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets("
        "const llvm::Triple &TT, ExceptionHandling ExceptionModel, "
        "FloatABI::ABIType FloatABI, EABI EABIVersion, "
        "StringRef ABIName) {\n"
        "  struct LibcallImplPair {\n"
        "    RTLIB::Libcall Func;\n"
        "    RTLIB::LibcallImpl Impl;\n"
        "  };\n"
        "  auto setLibcallsImpl = [this](\n"
        "    ArrayRef<LibcallImplPair> Libcalls,\n"
        "    std::optional<llvm::CallingConv::ID> CC = {})\n"
        "  {\n"
        "    for (const auto [Func, Impl] : Libcalls) {\n"
        "      setLibcallImpl(Func, Impl);\n"
        "      if (CC)\n"
        "        setLibcallImplCallingConv(Impl, *CC);\n"
        "    }\n"
        "  };\n";
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
                                         *this, Func2Preds));

    const SetTheory::RecVec *Elements =
        Sets.expand(R->getValueAsDef("MemberList"));

    // Sort to get deterministic output
    SetVector<PredicateWithCC> PredicateSorter;
    PredicateSorter.insert(
        PredicateWithCC()); // No predicate or CC override first.

    DenseMap<PredicateWithCC, LibcallsWithCC> Pred2Funcs;
    for (const Record *Elt : *Elements) {
      const RuntimeLibcallImpl *LibCallImpl = getRuntimeLibcallImpl(Elt);
      if (!LibCallImpl) {
        PrintError(R, "entry for SystemLibrary is not a RuntimeLibcallImpl");
        PrintNote(Elt->getLoc(), "invalid entry `" + Elt->getName() + "`");
        continue;
      }

      auto It = Func2Preds.find(LibCallImpl);
      if (It == Func2Preds.end()) {
        Pred2Funcs[PredicateWithCC()].LibcallImpls.push_back(LibCallImpl);
        continue;
      }

      for (const Record *Pred : It->second.first) {
        const Record *CC = It->second.second;
        PredicateWithCC Key(Pred, CC);

        auto &Entry = Pred2Funcs[Key];
        Entry.LibcallImpls.push_back(LibCallImpl);
        Entry.CallingConv = It->second.second;
        PredicateSorter.insert(Key);
      }
    }

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

      OS << indent(IndentDepth + 2) << "setLibcallsImpl({\n";
      for (const RuntimeLibcallImpl *LibCallImpl : Funcs) {
        OS << indent(IndentDepth + 4);
        LibCallImpl->emitTableEntry(OS);
      }
      OS << indent(IndentDepth + 2) << "}";
      if (FuncsWithCC.CallingConv) {
        StringRef CCEnum =
            FuncsWithCC.CallingConv->getValueAsString("CallingConv");
        OS << ", " << CCEnum;
      }
      OS << ");\n\n";

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

  OS << "#ifdef GET_SET_TARGET_RUNTIME_LIBCALL_SETS\n";
  emitSystemRuntimeLibrarySetCalls(OS);
  OS << "#endif\n\n";
}

void LibcallPredicateExpander::expand(SetTheory &ST, const Record *Def,
                                      SetTheory::RecSet &Elts) {
  assert(Def->isSubClassOf("LibcallImpls"));

  SetTheory::RecSet TmpElts;

  ST.evaluate(Def->getValueInit("MemberList"), TmpElts, Def->getLoc());

  Elts.insert(TmpElts.begin(), TmpElts.end());

  AvailabilityPredicate AP(Def->getValueAsDef("AvailabilityPredicate"));
  const Record *CCClass = Def->getValueAsOptionalDef("CallingConv");

  // This is assuming we aren't conditionally applying a calling convention to
  // some subsets, and not another, but this doesn't appear to be used.

  for (const Record *LibcallImplDef : TmpElts) {
    const RuntimeLibcallImpl *LibcallImpl =
        LibcallEmitter.getRuntimeLibcallImpl(LibcallImplDef);
    if (!AP.isAlwaysAvailable() || CCClass) {
      auto [It, Inserted] = Func2Preds.insert({LibcallImpl, {{}, CCClass}});
      if (!Inserted) {
        PrintError(
            Def,
            "combining nested libcall set predicates currently unhandled: '" +
                LibcallImpl->getLibcallFuncName() + "'");
      }

      It->second.first.push_back(AP.getDef());
      It->second.second = CCClass;
    }
  }
}

static TableGen::Emitter::OptClass<RuntimeLibcallEmitter>
    X("gen-runtime-libcalls", "Generate RuntimeLibcalls");
