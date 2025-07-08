//===- RuntimeLibcallEmitter.cpp - Properties from RuntimeLibcalls.td -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/SetTheory.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

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
};

class RuntimeLibcallEmitter;
class RuntimeLibcallImpl;

/// Used to apply predicates to nested sets of libcalls.
struct LibcallPredicateExpander : SetTheory::Expander {
  const RuntimeLibcallEmitter &LibcallEmitter;
  DenseMap<const Record *, std::vector<const Record *>> &Func2Preds;

  LibcallPredicateExpander(
      const RuntimeLibcallEmitter &LibcallEmitter,
      DenseMap<const Record *, std::vector<const Record *>> &Func2Preds)
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

  void emitQuotedLibcallFuncName(raw_ostream &OS) const {
    OS << '\"' << getLibcallFuncName() << '\"';
  }

  bool isDefault() const { return TheDef->getValueAsBit("IsDefault"); }

  void emitEnumEntry(raw_ostream &OS) const {
    OS << "RTLIB::" << TheDef->getName();
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

    ArrayRef<const Record *> AllRuntimeLibcallImpls =
        Records.getAllDerivedDefinitions("RuntimeLibcallImpl");
    RuntimeLibcallImplDefList.reserve(AllRuntimeLibcallImpls.size());

    size_t LibCallImplEnumVal = 1;
    for (const Record *LibCallImplDef : AllRuntimeLibcallImpls) {
      RuntimeLibcallImplDefList.emplace_back(LibCallImplDef, Def2RuntimeLibcall,
                                             LibCallImplEnumVal++);

      RuntimeLibcallImpl &LibCallImpl = RuntimeLibcallImplDefList.back();

      Def2RuntimeLibcallImpl[LibCallImplDef] = &LibCallImpl;

      // const RuntimeLibcallImpl &LibCallImpl =
      // RuntimeLibcallImplDefList.back();
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

  // TODO: Emit libcall names as string offset table.

  OS << "  UNKNOWN_LIBCALL = " << RuntimeLibcallDefList.size()
     << "\n};\n\n"
        "enum LibcallImpl : unsigned short {\n"
        "  Unsupported = 0,\n";

  // FIXME: Emit this in a different namespace. And maybe use enum class.
  for (const RuntimeLibcallImpl &LibCall : RuntimeLibcallImplDefList) {
    OS << "  " << LibCall.getName() << " = " << LibCall.getEnumVal() << ", // "
       << LibCall.getLibcallFuncName() << '\n';
  }

  OS << "  NumLibcallImpls = " << RuntimeLibcallImplDefList.size() + 1
     << "\n};\n"
        "} // End namespace RTLIB\n"
        "} // End namespace llvm\n"
        "#endif\n\n";
}

void RuntimeLibcallEmitter::emitGetInitRuntimeLibcallNames(
    raw_ostream &OS) const {
  // TODO: Emit libcall names as string offset table.

  OS << "const RTLIB::LibcallImpl "
        "llvm::RTLIB::RuntimeLibcallsInfo::"
        "DefaultLibcallImpls[RTLIB::UNKNOWN_LIBCALL + 1] = {\n";

  for (const RuntimeLibcall &LibCall : RuntimeLibcallDefList) {
    auto I = LibCallToDefaultImpl.find(&LibCall);
    if (I == LibCallToDefaultImpl.end()) {
      OS << "  RTLIB::Unsupported,";
    } else {
      const RuntimeLibcallImpl *LibCallImpl = I->second;
      OS << "  ";
      LibCallImpl->emitEnumEntry(OS);
      OS << ',';
    }

    OS << " // ";
    LibCall.emitEnumEntry(OS);
    OS << '\n';
  }

  OS << "  RTLIB::Unsupported\n"
        "};\n\n";

  // Emit the implementation names
  OS << "const char *const llvm::RTLIB::RuntimeLibcallsInfo::"
        "LibCallImplNames[RTLIB::NumLibcallImpls] = {\n"
        "  nullptr, // RTLIB::Unsupported\n";

  for (const RuntimeLibcallImpl &LibCallImpl : RuntimeLibcallImplDefList) {
    OS << "  \"" << LibCallImpl.getLibcallFuncName() << "\", // ";
    LibCallImpl.emitEnumEntry(OS);
    OS << '\n';
  }

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
}

void RuntimeLibcallEmitter::emitSystemRuntimeLibrarySetCalls(
    raw_ostream &OS) const {
  OS << "void llvm::RTLIB::RuntimeLibcallsInfo::setTargetRuntimeLibcallSets("
        "const llvm::Triple &TT) {\n"
        "  struct LibcallImplPair {\n"
        "    RTLIB::Libcall Func;\n"
        "    RTLIB::LibcallImpl Impl;\n"
        "  };\n";
  ArrayRef<const Record *> AllLibs =
      Records.getAllDerivedDefinitions("SystemRuntimeLibrary");

  for (const Record *R : AllLibs) {
    OS << '\n';

    AvailabilityPredicate TopLevelPredicate(R->getValueAsDef("TriplePred"));

    OS << indent(2);
    TopLevelPredicate.emitIf(OS);
    SetTheory Sets;

    DenseMap<const Record *, std::vector<const Record *>> Func2Preds;
    Sets.addExpander("LibcallImpls", std::make_unique<LibcallPredicateExpander>(
                                         *this, Func2Preds));

    const SetTheory::RecVec *Elements =
        Sets.expand(R->getValueAsDef("MemberList"));

    // Sort to get deterministic output
    SetVector<const Record *> PredicateSorter;
    PredicateSorter.insert(nullptr); // No predicate first.

    DenseMap<const Record *, std::vector<const RuntimeLibcallImpl *>>
        Pred2Funcs;
    for (const Record *Elt : *Elements) {
      const RuntimeLibcallImpl *LibCallImpl = getRuntimeLibcallImpl(Elt);
      if (!LibCallImpl) {
        PrintError(R, "entry for SystemLibrary is not a RuntimeLibcallImpl");
        PrintNote(Elt->getLoc(), "invalid entry `" + Elt->getName() + "`");
        continue;
      }

      auto It = Func2Preds.find(Elt);
      if (It == Func2Preds.end()) {
        Pred2Funcs[nullptr].push_back(LibCallImpl);
        continue;
      }

      for (const Record *Pred : It->second) {
        Pred2Funcs[Pred].push_back(LibCallImpl);
        PredicateSorter.insert(Pred);
      }
    }

    SmallVector<const Record *, 0> SortedPredicates =
        PredicateSorter.takeVector();

    sort(SortedPredicates, [](const Record *A, const Record *B) {
      if (!A)
        return true;
      if (!B)
        return false;
      return A->getName() < B->getName();
    });

    for (const Record *Pred : SortedPredicates) {
      AvailabilityPredicate SubsetPredicate(Pred);
      unsigned IndentDepth = 2;

      auto It = Pred2Funcs.find(Pred);
      if (It == Pred2Funcs.end())
        continue;

      if (!SubsetPredicate.isAlwaysAvailable()) {
        IndentDepth = 4;

        OS << indent(IndentDepth);
        SubsetPredicate.emitIf(OS);
      }

      std::vector<const RuntimeLibcallImpl *> &Funcs = It->second;

      // Ensure we only emit a unique implementation per libcall in the
      // selection table.
      //
      // FIXME: We need to generate separate functions for
      // is-libcall-available and should-libcall-be-used to avoid this.
      //
      // This also makes it annoying to make use of the default set, since the
      // entries from the default set may win over the replacements unless
      // they are explicitly removed.
      sort(Funcs, [](const RuntimeLibcallImpl *A, const RuntimeLibcallImpl *B) {
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
         << "static const LibcallImplPair LibraryCalls[] = {\n";
      for (const RuntimeLibcallImpl *LibCallImpl : Funcs) {
        OS << indent(IndentDepth + 6);
        LibCallImpl->emitTableEntry(OS);
      }

      OS << indent(IndentDepth + 2) << "};\n\n"
         << indent(IndentDepth + 2)
         << "for (const auto [Func, Impl] : LibraryCalls) {\n"
         << indent(IndentDepth + 2) << "  setLibcallImpl(Func, Impl);\n"
         << indent(IndentDepth + 2) << "}\n";

      if (!SubsetPredicate.isAlwaysAvailable()) {
        OS << indent(IndentDepth);
        SubsetPredicate.emitEndIf(OS);
        OS << '\n';
      }
    }

    OS << indent(4) << "return;\n" << indent(2);
    TopLevelPredicate.emitEndIf(OS);
  }

  // Fallback to the old default set for manual table entries.
  //
  // TODO: Remove this when targets have switched to using generated tables by
  // default.
  OS << "  initDefaultLibCallImpls();\n";

  OS << "}\n\n";
}

void RuntimeLibcallEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Runtime LibCalls Source Fragment", OS, Records);
  emitGetRuntimeLibcallEnum(OS);

  OS << "#ifdef GET_INIT_RUNTIME_LIBCALL_NAMES\n";
  emitGetInitRuntimeLibcallNames(OS);
  OS << "#endif\n\n";

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

  for (const Record *LibcallImpl : TmpElts) {
    if (!AP.isAlwaysAvailable()) {
      auto [It, Inserted] = Func2Preds.insert({LibcallImpl, {}});
      if (!Inserted) {
        PrintError(
            Def, "combining nested libcall set predicates currently unhandled");
      }

      It->second.push_back(AP.getDef());
    }
  }
}

static TableGen::Emitter::OptClass<RuntimeLibcallEmitter>
    X("gen-runtime-libcalls", "Generate RuntimeLibcalls");
