#include "llvm/ADT/SmallString.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <vector>

using namespace llvm;

namespace {
class IntrinsicFolderEmitter {
  const RecordKeeper &Records;

public:
  IntrinsicFolderEmitter(const RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS) {
    emitSwitchFunction(OS);
  }

private:
  void emitSwitchFunction(raw_ostream &OS) {
    const char *ConstRecordName = "ConstFolder";
    ArrayRef<const Record*> Intrinsics =
        Records.getAllDerivedDefinitions("Intrinsic");

    OS << "// Automatically generated file. Do not edit!\n";
    OS << "// Intrinsic constant folding switch function\n\n";

    OS << "static Constant *foldIntrinsic(StringRef Name, Intrinsic::ID IntrinsicID,\n";
    OS << "                               Type *Ty, ArrayRef<Constant *> Operands,\n";
    OS << "                               const DataLayout &DL, const TargetLibraryInfo *TLI,\n";
    OS << "                               const CallBase *Call) {\n";
    OS << "  switch (IntrinsicID) {\n";
    OS << "    default: return nullptr;\n";

    for (const Record *R : Intrinsics) {
      if (R->isValueUnset(ConstRecordName))
        continue;

      StringRef FolderName = R->getValueAsString(ConstRecordName);
      if (FolderName.empty())
        continue;

      StringRef EnumName = R->getName();
      OS << "    case Intrinsic::" << EnumName << ":\n";
      OS << "      return " << FolderName << "(Name, Ty, Operands, DL, TLI, Call);\n";
    }

    OS << "  }\n";
    OS << "}\n";
  }
};
} // namespace

static TableGen::Emitter::OptClass<IntrinsicFolderEmitter>
    X("gen-constant-folding", "Generate functions for constants folding for intrinsics");