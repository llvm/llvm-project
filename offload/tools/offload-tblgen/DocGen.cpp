//===- offload-tblgen/DocGen.cpp - Tablegen backend for Offload header ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Tablegen backend that produces the contents of the Offload API
// specification. The generated reStructuredText is Sphinx compatible, see
// https://www.sphinx-doc.org/en/master/usage/domains/c.html for further
// details on the C language domain.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "GenCommon.hpp"
#include "RecordTypes.hpp"

using namespace llvm;
using namespace offload::tblgen;

namespace {
std::string makeFunctionSignature(StringRef RetTy, StringRef Name,
                                  ArrayRef<ParamRec> Params) {
  std::string S;
  raw_string_ostream OS{S};
  OS << RetTy << " " << Name << "(";
  for (const ParamRec &Param : Params) {
    OS << Param.getType() << " " << Param.getName();
    if (Param != Params.back()) {
      OS << ", ";
    }
  }
  OS << ")";
  return S;
}

std::string makeDoubleBackticks(StringRef R) {
  std::string S;
  for (char C : R) {
    if (C == '`') {
      S.push_back('`');
    }
    S.push_back(C);
  }
  return S;
}

void processMacro(const MacroRec &M, raw_ostream &OS) {
  OS << formatv(".. c:macro:: {0}\n\n", M.getNameWithArgs());
  OS << "  " << M.getDesc() << "\n\n";
}

void processTypedef(const TypedefRec &T, raw_ostream &OS) {
  OS << formatv(".. c:type:: {0} {1}\n\n", T.getValue(), T.getName());
  OS << "  " << T.getDesc() << "\n\n";
}

void processHandle(const HandleRec &H, raw_ostream &OS) {

  OS << formatv(".. c:type:: struct {0} *{1}\n\n", getHandleImplName(H),
                H.getName());
  OS << "  " << H.getDesc() << "\n\n";
}

void processFptrTypedef(const FptrTypedefRec &F, raw_ostream &OS) {
  OS << ".. c:type:: "
     << makeFunctionSignature(F.getReturn(),
                              StringRef{formatv("(*{0})", F.getName())},
                              F.getParams())
     << "\n\n";
  for (const ParamRec &P : F.getParams()) {
    OS << formatv("  :param {0}: {1}\n", P.getName(), P.getDesc());
  }
  OS << "\n";
}

void processEnum(const EnumRec &E, raw_ostream &OS) {
  OS << formatv(".. c:enum:: {0}\n\n", E.getName());
  OS << "  " << E.getDesc() << "\n\n";
  for (const EnumValueRec Etor : E.getValues()) {
    OS << formatv("  .. c:enumerator:: {0}_{1}\n\n", E.getEnumValNamePrefix(),
                  Etor.getName());
    OS << "    ";
    if (E.isTyped()) {
      OS << ":c:expr:`" << Etor.getTaggedType() << "` — ";
    }
    OS << Etor.getDesc() << "\n\n";
  }
}

void processStruct(const StructRec &S, raw_ostream &OS) {
  OS << formatv(".. c:struct:: {0}\n\n", S.getName());
  OS << "  " << S.getDesc() << "\n\n";
  for (const StructMemberRec &M : S.getMembers()) {
    OS << formatv("  .. c:member:: {0} {1}\n\n", M.getType(), M.getName());
    OS << "    " << M.getDesc() << "\n\n";
  }
}

void processFunction(const FunctionRec &F, raw_ostream &OS) {
  OS << ".. c:function:: "
     << makeFunctionSignature({formatv("{0}_result_t", PrefixLower)},
                              F.getName(), F.getParams())
     << "\n\n";

  OS << "  " << F.getDesc() << "\n\n";
  for (StringRef D : F.getDetails()) {
    OS << "  " << D << "\n";
  }
  if (!F.getDetails().empty()) {
    OS << "\n";
  }

  for (const ParamRec &P : F.getParams()) {
    OS << formatv("  :param {0}: {1}\n", P.getName(), P.getDesc());
  }

  for (const ReturnRec &R : F.getReturns()) {
    OS << formatv("  :retval {0}:\n", R.getValue());
    for (StringRef C : R.getConditions()) {
      OS << "    * ";
      if (C.starts_with("`") && C.ends_with("`")) {
        OS << ":c:expr:" << C;
      } else {
        OS << makeDoubleBackticks(C);
      }
      OS << "\n";
    }
  }
  OS << "\n";
}
} // namespace

void EmitOffloadDoc(const RecordKeeper &Records, raw_ostream &OS) {
  OS << "Offload API\n";
  OS << "===========\n\n";

  ArrayRef<const Record *> Macros = Records.getAllDerivedDefinitions("Macro");
  if (!Macros.empty()) {
    OS << "Macros\n";
    OS << "------\n\n";
    for (const Record *M : Macros) {
      processMacro(MacroRec{M}, OS);
    }
  }

  ArrayRef<const Record *> Handles = Records.getAllDerivedDefinitions("Handle");
  ArrayRef<const Record *> Typedefs =
      Records.getAllDerivedDefinitions("Typedef");
  ArrayRef<const Record *> FptrTypedefs =
      Records.getAllDerivedDefinitions("FptrTypedef");
  if (!Handles.empty() || !Typedefs.empty() || !FptrTypedefs.empty()) {
    OS << "Type Definitions\n";
    OS << "----------------\n\n";
    for (const Record *H : Handles) {
      processHandle(HandleRec{H}, OS);
    }
    for (const Record *T : Typedefs) {
      processTypedef(TypedefRec{T}, OS);
    }
    for (const Record *F : FptrTypedefs) {
      processFptrTypedef(FptrTypedefRec{F}, OS);
    }
  }

  ArrayRef<const Record *> Enums = Records.getAllDerivedDefinitions("Enum");
  OS << "Enums\n";
  OS << "-----\n\n";
  if (!Enums.empty()) {
    for (const Record *E : Enums) {
      processEnum(EnumRec{E}, OS);
    }
  }

  ArrayRef<const Record *> Structs = Records.getAllDerivedDefinitions("Struct");
  if (!Structs.empty()) {
    OS << "Structs\n";
    OS << "-------\n\n";
    for (const Record *S : Structs) {
      processStruct(StructRec{S}, OS);
    }
  }

  ArrayRef<const Record *> Functions =
      Records.getAllDerivedDefinitions("Function");
  if (!Functions.empty()) {
    OS << "Functions\n";
    OS << "---------\n\n";
    for (const Record *F : Functions) {
      processFunction(FunctionRec{F}, OS);
    }
  }
}
