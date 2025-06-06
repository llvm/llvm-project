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
#include "llvm/TableGen/TableGenBackend.h"
using namespace llvm;

namespace {

class RuntimeLibcall {
  const Record *TheDef = nullptr;

public:
  RuntimeLibcall() = delete;
  RuntimeLibcall(const Record *Def) : TheDef(Def) { assert(Def); }

  ~RuntimeLibcall() { assert(TheDef); }

  const Record *getDef() const { return TheDef; }

  StringRef getName() const { return TheDef->getName(); }

  void emitEnumEntry(raw_ostream &OS) const {
    OS << "RTLIB::" << TheDef->getValueAsString("Name");
  }
};

class RuntimeLibcallImpl {
  const Record *TheDef;
  const RuntimeLibcall *Provides = nullptr;

public:
  RuntimeLibcallImpl(
      const Record *Def,
      const DenseMap<const Record *, const RuntimeLibcall *> &ProvideMap)
      : TheDef(Def) {
    if (const Record *ProvidesDef = Def->getValueAsDef("Provides"))
      Provides = ProvideMap.lookup(ProvidesDef);
  }

  ~RuntimeLibcallImpl() {}

  const Record *getDef() const { return TheDef; }

  StringRef getName() const { return TheDef->getName(); }

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
};

class RuntimeLibcallEmitter {
private:
  const RecordKeeper &Records;

  DenseMap<const Record *, const RuntimeLibcall *> Def2RuntimeLibcall;

  const RuntimeLibcall *getRuntimeLibcall(const Record *Def) const {
    return Def2RuntimeLibcall.lookup(Def);
  }

  std::vector<RuntimeLibcall> RuntimeLibcallDefList;
  std::vector<RuntimeLibcallImpl> RuntimeLibcallImplDefList;

  DenseMap<const RuntimeLibcall *, const RuntimeLibcallImpl *>
      LibCallToDefaultImpl;

  void
  emitTargetOverrideFunc(raw_ostream &OS, StringRef FuncName,
                         ArrayRef<RuntimeLibcallImpl> LibCallImplList) const;

  void emitGetRuntimeLibcallEnum(raw_ostream &OS) const;

  void emitWindowsArm64LibCallNameOverrides(raw_ostream &OS) const;

  void emitGetInitRuntimeLibcallNames(raw_ostream &OS) const;
  void emitGetInitRuntimeLibcallUtils(raw_ostream &OS) const;

public:
  RuntimeLibcallEmitter(const RecordKeeper &R) : Records(R) {

    ArrayRef<const Record *> AllRuntimeLibcalls =
        Records.getAllDerivedDefinitions("RuntimeLibcall");

    RuntimeLibcallDefList.reserve(AllRuntimeLibcalls.size());

    for (const Record *RuntimeLibcallDef : AllRuntimeLibcalls) {
      RuntimeLibcallDefList.emplace_back(RuntimeLibcallDef);
      Def2RuntimeLibcall[RuntimeLibcallDef] = &RuntimeLibcallDefList.back();
    }

    for (RuntimeLibcall &LibCall : RuntimeLibcallDefList)
      Def2RuntimeLibcall[LibCall.getDef()] = &LibCall;

    ArrayRef<const Record *> AllRuntimeLibcallImpls =
        Records.getAllDerivedDefinitions("RuntimeLibcallImpl");
    RuntimeLibcallImplDefList.reserve(AllRuntimeLibcallImpls.size());

    for (const Record *LibCallImplDef : AllRuntimeLibcallImpls) {
      RuntimeLibcallImplDefList.emplace_back(LibCallImplDef,
                                             Def2RuntimeLibcall);

      RuntimeLibcallImpl &LibCallImpl = RuntimeLibcallImplDefList.back();

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

  std::vector<RuntimeLibcallImpl>
  getRuntimeLibcallImplSet(StringRef Name) const {
    std::vector<RuntimeLibcallImpl> Result;
    ArrayRef<const Record *> ImplSet =
        Records.getAllDerivedDefinitionsIfDefined(Name);
    Result.reserve(ImplSet.size());

    for (const Record *LibCallImplDef : ImplSet)
      Result.emplace_back(LibCallImplDef, Def2RuntimeLibcall);
    return Result;
  }

  void run(raw_ostream &OS);
};

} // End anonymous namespace.

/// Emit a method \p FuncName of RTLIB::RuntimeLibcallsInfo to override the
/// libcall names in \p LibCallImplList.
void RuntimeLibcallEmitter::emitTargetOverrideFunc(
    raw_ostream &OS, StringRef FuncName,
    ArrayRef<RuntimeLibcallImpl> LibCallImplList) const {
  OS << "void llvm::RTLIB::RuntimeLibcallsInfo::" << FuncName << "() {\n";

  if (LibCallImplList.empty()) {
    OS << "  llvm_unreachable(\"override set not defined\");\n";
  } else {
    // for (const Record *LibCallImpl : LibCallImplList) {
    for (const RuntimeLibcallImpl &LibCallImpl : LibCallImplList) {
      const RuntimeLibcall *Provides = LibCallImpl.getProvides();
      OS << "  LibcallImpls[";
      Provides->emitEnumEntry(OS);
      OS << "] = ";
      LibCallImpl.emitEnumEntry(OS);
      OS << ";\n";
    }
  }

  OS << "}\n\n";
}

void RuntimeLibcallEmitter::emitGetRuntimeLibcallEnum(raw_ostream &OS) const {
  OS << "#ifdef GET_RUNTIME_LIBCALL_ENUM\n"
        "namespace llvm {\n"
        "namespace RTLIB {\n"
        "enum Libcall : unsigned short {\n";

  size_t CallTypeEnumVal = 0;
  for (const RuntimeLibcall &LibCall : RuntimeLibcallDefList) {
    StringRef Name = LibCall.getName();
    OS << "  " << Name << " = " << CallTypeEnumVal++ << ",\n";
  }

  // TODO: Emit libcall names as string offset table.

  OS << "  UNKNOWN_LIBCALL = " << CallTypeEnumVal
     << "\n};\n\n"
        "enum LibcallImpl : unsigned short {\n"
        "  Unsupported = 0,\n";

  // FIXME: Emit this in a different namespace. And maybe use enum class.
  size_t LibCallImplEnumVal = 1;
  for (const RuntimeLibcallImpl &LibCall : RuntimeLibcallImplDefList) {
    OS << "  " << LibCall.getName() << " = " << LibCallImplEnumVal++ << ", // "
       << LibCall.getLibcallFuncName() << '\n';
  }

  OS << "  NumLibcallImpls = " << LibCallImplEnumVal
     << "\n};\n"
        "} // End namespace RTLIB\n"
        "} // End namespace llvm\n"
        "#endif\n\n";
}

void RuntimeLibcallEmitter::emitWindowsArm64LibCallNameOverrides(
    raw_ostream &OS) const {
  // FIXME: Stop treating this as a special case
  OS << "void "
        "llvm::RTLIB::RuntimeLibcallsInfo::setWindowsArm64LibCallNameOverrides("
        ") {\n"
        "  static const RTLIB::LibcallImpl "
        "WindowsArm64RoutineImpls[RTLIB::UNKNOWN_LIBCALL + 1] = {\n";
  for (const RuntimeLibcall &LibCall : RuntimeLibcallDefList) {
    auto I = LibCallToDefaultImpl.find(&LibCall);
    if (I == LibCallToDefaultImpl.end())
      OS << "    RTLIB::Unsupported,";
    else {
      const RuntimeLibcallImpl *LibCallImpl = I->second;
      assert(LibCallImpl);
      OS << "    RTLIB::arm64ec_" << LibCallImpl->getName() << ',';
    }

    OS << " // ";
    LibCall.emitEnumEntry(OS);
    OS << '\n';
  }

  OS << "    RTLIB::Unsupported // RTLIB::UNKNOWN_LIBCALL\n"
        "  };\n\n"
        "  std::memcpy(LibcallImpls, WindowsArm64RoutineImpls,\n"
        "              sizeof(LibcallImpls));\n"
        "  static_assert(sizeof(LibcallImpls) == "
        "sizeof(WindowsArm64RoutineImpls),\n"
        "                \"libcall array size should match\");\n"
        "}\n#endif\n\n";
}

void RuntimeLibcallEmitter::emitGetInitRuntimeLibcallNames(
    raw_ostream &OS) const {
  // TODO: Emit libcall names as string offset table.

  OS << "#ifdef GET_INIT_RUNTIME_LIBCALL_NAMES\n"
        "const RTLIB::LibcallImpl "
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
      OS << ",";
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

  std::vector<RuntimeLibcallImpl> ZOSRuntimeLibcallImplList =
      getRuntimeLibcallImplSet("ZOSRuntimeLibcallImpl");
  emitTargetOverrideFunc(OS, "setZOSLibCallNameOverrides",
                         ZOSRuntimeLibcallImplList);

  std::vector<RuntimeLibcallImpl> PPCRuntimeLibcallImplList =
      getRuntimeLibcallImplSet("PPCRuntimeLibcallImpl");
  emitTargetOverrideFunc(OS, "setPPCLibCallNameOverrides",
                         PPCRuntimeLibcallImplList);

  emitWindowsArm64LibCallNameOverrides(OS);
}

void RuntimeLibcallEmitter::emitGetInitRuntimeLibcallUtils(
    raw_ostream &OS) const {
  // FIXME: Hack we shouldn't really need
  OS << "#ifdef GET_INIT_RUNTIME_LIBCALL_UTILS\n"
        "static inline bool isAtomicLibCall(llvm::RTLIB::Libcall LC) {\n"
        "  switch (LC) {\n";
  for (const RuntimeLibcall &LibCall : RuntimeLibcallDefList) {
    StringRef Name = LibCall.getName();
    if (Name.contains("ATOMIC")) {
      OS << "  case ";
      LibCall.emitEnumEntry(OS);
      OS << ":\n";
    }
  }

  OS << "    return true;\n"
        "  default:\n"
        "    return false;\n"
        "  }\n\n"
        "  llvm_unreachable(\"covered switch over libcalls\");\n"
        "}\n#endif\n\n";
}

void RuntimeLibcallEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Runtime LibCalls Source Fragment", OS, Records);
  emitGetRuntimeLibcallEnum(OS);
  emitGetInitRuntimeLibcallNames(OS);
  emitGetInitRuntimeLibcallUtils(OS);
}

static TableGen::Emitter::OptClass<RuntimeLibcallEmitter>
    X("gen-runtime-libcalls", "Generate RuntimeLibcalls");
