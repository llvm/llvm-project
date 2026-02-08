#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

// utils/TableGen/IntrinsicFolderEmitter.cpp
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
    // Get all intrinsic definitions
    ArrayRef<const Record*> Intrinsics =
        Records.getAllDerivedDefinitions("Intrinsic");

    OS << "// Automatically generated file. Do not edit!\n";
    OS << "// Intrinsic constant folding switch function\n\n";

    OS << "static Constant *foldIntrinsic(Intrinsic::ID IntrinsicID, Type *Ty,\n";
    OS << "                               ArrayRef<Constant *> Operands,\n";
    OS << "                               const CallBase *Call) {\n";
    OS << "  switch (IntrinsicID) {\n";
    OS << "    default: return nullptr;\n";

    for (const Record *R : Intrinsics) {
      if (R->isValueUnset("Folder"))
        continue;

      StringRef FolderName = R->getValueAsString("Folder");
      if (FolderName.empty())
        continue;

      StringRef EnumName = R->getName();
      // TODO: make a proper interface
      OS << "    case Intrinsic::" << EnumName << ":\n";
      OS << "      return " << FolderName << "(...);\n";
    }

    OS << "  }\n";
    OS << "}\n";
  }
};
} // namespace

static TableGen::Emitter::OptClass<IntrinsicFolderEmitter>
    X("gen-constant-folding-functions", "Generate functions for constants folding for intrinsics");