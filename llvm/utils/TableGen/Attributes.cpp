//===- Attributes.cpp - Generate attributes -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <string>
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "attr-enum"

namespace {

class Attributes {
public:
  Attributes(RecordKeeper &R) : Records(R) {}
  void emit(raw_ostream &OS);

private:
  void emitTargetIndependentNames(raw_ostream &OS);
  void emitFnAttrCompatCheck(raw_ostream &OS, bool IsStringAttr);

  RecordKeeper &Records;
};

} // End anonymous namespace.

void Attributes::emitTargetIndependentNames(raw_ostream &OS) {
  OS << "#ifdef GET_ATTR_NAMES\n";
  OS << "#undef GET_ATTR_NAMES\n";

  OS << "#ifndef ATTRIBUTE_ALL\n";
  OS << "#define ATTRIBUTE_ALL(FIRST, SECOND)\n";
  OS << "#endif\n\n";

  auto Emit = [&](ArrayRef<StringRef> KindNames, StringRef MacroName) {
    OS << "#ifndef " << MacroName << "\n";
    OS << "#define " << MacroName
       << "(FIRST, SECOND) ATTRIBUTE_ALL(FIRST, SECOND)\n";
    OS << "#endif\n\n";
    for (StringRef KindName : KindNames) {
      for (auto A : Records.getAllDerivedDefinitions(KindName)) {
        OS << MacroName << "(" << A->getName() << ","
           << A->getValueAsString("AttrString") << ")\n";
      }
    }
    OS << "#undef " << MacroName << "\n\n";
  };

  // Emit attribute enums in the same order llvm::Attribute::operator< expects.
  Emit({"EnumAttr", "TypeAttr", "IntAttr"}, "ATTRIBUTE_ENUM");
  Emit({"StrBoolAttr"}, "ATTRIBUTE_STRBOOL");

  OS << "#undef ATTRIBUTE_ALL\n";
  OS << "#endif\n";
}

void Attributes::emitFnAttrCompatCheck(raw_ostream &OS, bool IsStringAttr) {
  OS << "#ifdef GET_ATTR_COMPAT_FUNC\n";
  OS << "#undef GET_ATTR_COMPAT_FUNC\n";

  OS << "static inline bool hasCompatibleFnAttrs(const Function &Caller,\n"
     << "                                        const Function &Callee) {\n";
  OS << "  bool Ret = true;\n\n";

  std::vector<Record *> CompatRules =
      Records.getAllDerivedDefinitions("CompatRule");

  for (auto *Rule : CompatRules) {
    StringRef FuncName = Rule->getValueAsString("CompatFunc");
    OS << "  Ret &= " << FuncName << "(Caller, Callee);\n";
  }

  OS << "\n";
  OS << "  return Ret;\n";
  OS << "}\n\n";

  std::vector<Record *> MergeRules =
      Records.getAllDerivedDefinitions("MergeRule");
  OS << "static inline void mergeFnAttrs(Function &Caller,\n"
     << "                                const Function &Callee) {\n";

  for (auto *Rule : MergeRules) {
    StringRef FuncName = Rule->getValueAsString("MergeFunc");
    OS << "  " << FuncName << "(Caller, Callee);\n";
  }

  OS << "}\n\n";

  OS << "#endif\n";
}

void Attributes::emit(raw_ostream &OS) {
  emitTargetIndependentNames(OS);
  emitFnAttrCompatCheck(OS, false);
}

namespace llvm {

void EmitAttributes(RecordKeeper &RK, raw_ostream &OS) {
  Attributes(RK).emit(OS);
}

} // End llvm namespace.
