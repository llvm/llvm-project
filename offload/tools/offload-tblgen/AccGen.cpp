//===- offload-tblgen/AccGen.cpp - OpenACC runtime API backend ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Auto generates the OpenACC API from the tablegen definitions.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include "GenCommon.hpp"
#include "Generators.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

namespace {

constexpr StringLiteral AccBindPrefix = "__olacc_f_";
constexpr StringLiteral ImplNamespace = "llvm::acc::impl::f";

struct Param {
  std::string C;
  std::string F;
  std::string Name;
  std::string Intent;
  std::string Print;
  bool Value = false;
  std::string IgnoreTKR;
};

struct Func {
  std::string Generic;
  std::string Specific;
  std::string Bind;
  std::string RetC;
  std::string RetF;
  std::string RetPrint;
  std::vector<std::string> Aliases;
  std::vector<Param> Params;

  bool subroutine() const { return RetC == "void"; }
};

struct SigParam {
  std::string Type;
  std::string Name;
};

} // namespace

static void emitTypeAndName(raw_ostream &OS, StringRef Type, StringRef Name) {
  StringRef Ty = Type.rtrim();
  OS << Ty;
  if (!Ty.ends_with("*"))
    OS << " ";
  if (!Name.empty())
    OS << Name;
}

static void emitSignature(raw_ostream &OS, StringRef Ret, StringRef Name,
                          ArrayRef<SigParam> Ps, bool Semi, bool Export) {
  if (Export)
    OS << "ACC_API_EXPORT ";
  emitTypeAndName(OS, Ret, Name);
  OS << "(";
  if (Ps.empty())
    OS << "void";
  else {
    for (size_t I = 0; I < Ps.size(); ++I) {
      if (I)
        OS << ", ";
      emitTypeAndName(OS, Ps[I].Type, Ps[I].Name);
    }
  }
  OS << ")";
  if (Semi)
    OS << ";\n";
  else
    OS << " {\n";
}

static std::vector<SigParam> cParams(const Func &F) {
  std::vector<SigParam> Ps;
  for (const Param &P : F.Params)
    Ps.push_back({P.C, P.Name});
  return Ps;
}

static Param loadParam(const Record *P) {
  const Record *Ty = P->getValueAsDef("type");
  Param A;
  A.C = Ty->getValueAsString("c_spelling").str();
  A.F = Ty->getValueAsString("f_spelling").str();
  A.Name = P->getValueAsString("name").str();
  A.Intent = P->getValueAsString("f_intent").str();
  A.Print = Ty->getValueAsString("print_fn").str();
  A.Value = P->getValueAsBit("f_value");
  A.IgnoreTKR = Ty->getValueAsString("ignore_tkr").str();
  if (A.F.empty())
    PrintFatalError(P, "a parameter needs a Fortran spelling");
  return A;
}

static Func loadFFunc(const Record *R) {
  const Record *Sig = R->getValueAsDef("signature");
  const Record *Ret = Sig->getValueAsDef("result");
  Func F;
  F.Generic = R->getValueAsString("name").str();
  if (F.Generic.empty())
    PrintFatalError(R, "AccFFunc needs a name");
  F.RetC = Ret->getValueAsString("c_spelling").str();
  F.RetF = Ret->getValueAsString("f_spelling").str();
  F.RetPrint = Ret->getValueAsString("print_fn").str();
  for (StringRef Alias : R->getValueAsListOfStrings("alias_names"))
    F.Aliases.push_back(Alias.str());
  for (const Record *P : Sig->getValueAsListOfDefs("params"))
    F.Params.push_back(loadParam(P));
  std::string Prefix = R->getValueAsString("prefix").str();
  F.Specific = Prefix + F.Generic;
  F.Bind = (AccBindPrefix + Prefix + F.Generic).str();
  if (!F.subroutine() && F.RetF.empty())
    PrintFatalError(R, "a Fortran function needs a result spelling");
  return F;
}

static std::vector<Func> collectFFuncs(const RecordKeeper &Records) {
  std::vector<Func> Funcs;
  for (const Record *R : Records.getAllDerivedDefinitions("AccFFunc"))
    Funcs.push_back(loadFFunc(R));
  llvm::sort(Funcs,
             [](const Func &A, const Func &B) { return A.Bind < B.Bind; });
  return Funcs;
}

static bool isImportName(StringRef Name) {
  if (Name.empty() || (!isAlpha(Name.front()) && Name.front() != '_'))
    return false;
  return all_of(Name.drop_front(),
                [](char C) { return isAlnum(C) || C == '_'; });
}

static void addImport(std::vector<std::string> &Imps, StringRef Spelling) {
  size_t L = Spelling.find('(');
  if (L == StringRef::npos)
    return;
  size_t R = Spelling.find(')', L);
  if (R == StringRef::npos || R <= L + 1)
    return;
  StringRef Name = Spelling.slice(L + 1, R).trim();
  if (!isImportName(Name) || is_contained(Imps, Name))
    return;
  Imps.push_back(Name.str());
}

static void emitFInterface(raw_ostream &OS, const Func &F) {
  std::string Head = "    ";
  if (!F.subroutine())
    Head += F.RetF + " ";
  Head += (F.subroutine() ? "subroutine " : "function ");
  Head += F.Specific + "(";
  for (size_t I = 0; I < F.Params.size(); ++I) {
    if (I)
      Head += ", ";
    Head += F.Params[I].Name;
  }
  Head += ")";
  OS << Head << " bind(c, name=\"" << F.Bind << "\")\n";

  std::vector<std::string> Imps;
  if (!F.subroutine())
    addImport(Imps, F.RetF);
  for (const Param &P : F.Params)
    addImport(Imps, P.F);
  if (!Imps.empty()) {
    OS << "      import :: " << Imps[0];
    for (size_t I = 1; I < Imps.size(); ++I)
      OS << ", " << Imps[I];
    OS << "\n";
  }
  for (const Param &P : F.Params) {
    if (!P.IgnoreTKR.empty())
      OS << "      !dir$ ignore_tkr (" << P.IgnoreTKR << ") " << P.Name << "\n";
    OS << "      " << P.F;
    if (P.Value)
      OS << ", value";
    if (!P.Intent.empty())
      OS << ", intent(" << P.Intent << ")";
    OS << " :: " << P.Name << "\n";
  }
  OS << "    end " << (F.subroutine() ? "subroutine" : "function") << "\n";
}

static void emitNames(raw_ostream &OS, ArrayRef<SigParam> Ps) {
  for (size_t I = 0; I < Ps.size(); ++I) {
    if (I)
      OS << ", ";
    OS << Ps[I].Name;
  }
}

/// Emit `Value`, wrapped in the type's print function if it has one.
static void emitPrinted(raw_ostream &OS, StringRef PrintFn, StringRef Value) {
  if (PrintFn.empty())
    OS << Value;
  else
    OS << "llvm::acc::print::" << PrintFn << "(" << Value << ")";
}

static void emitArgTrace(raw_ostream &OS, const Func &F) {
  OS << TAB_1 << "ODBG(ADT_APITrace) << \"enter " << F.Specific << "(\"";
  for (size_t I = 0; I < F.Params.size(); ++I) {
    if (I > 0)
      OS << " << \", \"";
    OS << " << ";
    emitPrinted(OS, F.Params[I].Print, F.Params[I].Name);
  }
  OS << " << \")\";\n";
}

void EmitAccFEntries(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader << "\n";
  for (const Func &F : collectFFuncs(Records)) {
    auto Ps = cParams(F);
    emitSignature(OS, F.RetC, F.Bind, Ps, false, true);
    emitArgTrace(OS, F);
    if (F.subroutine()) {
      OS << TAB_1 << ImplNamespace << "::" << F.Specific << "(";
      emitNames(OS, Ps);
      OS << ");\n";
      OS << TAB_1 << "ODBG(ADT_APITrace) << \"exit  " << F.Specific << "\";\n";
    } else {
      OS << TAB_1;
      emitTypeAndName(OS, F.RetC, "res");
      OS << " = " << ImplNamespace << "::" << F.Specific << "(";
      emitNames(OS, Ps);
      OS << ");\n";
      OS << TAB_1 << "ODBG(ADT_APITrace) << \"exit  " << F.Specific
         << " -> \" << ";
      emitPrinted(OS, F.RetPrint, "res");
      OS << ";\n";
      OS << TAB_1 << "return res;\n";
    }
    OS << "}\n\n";
  }
}

void EmitAccFImplDecls(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader << "\n";
  for (const Func &F : collectFFuncs(Records))
    emitSignature(OS, F.RetC, F.Specific, cParams(F), true, false);
}

void EmitAccFImpls(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader << "\n";
  for (const Func &F : collectFFuncs(Records)) {
    emitSignature(OS, F.RetC, F.Specific, cParams(F), false, false);
    OS << TAB_1 << "ACC_API_NYI();\n";
    OS << "}\n\n";
  }
}

static StringRef recName(const Record *R) {
  StringRef Name = R->getValueAsString("name");
  if (Name.empty())
    PrintFatalError(R, "needs a non-empty name");
  return Name;
}

void EmitAccFModule(const RecordKeeper &Records, raw_ostream &OS) {
  for (const Record *R : Records.getAllDerivedDefinitions("FortranInt"))
    OS << "  integer, parameter :: " << recName(R) << " = "
       << R->getValueAsInt("value") << "\n";
  for (const Record *R : Records.getAllDerivedDefinitions("FortranKind"))
    OS << "  integer, parameter :: " << recName(R) << " = "
       << R->getValueAsString("value") << "\n";
  OS << "\n";
  for (const Record *R : Records.getAllDerivedDefinitions("AccEnum")) {
    StringRef KindName = recName(R->getValueAsDef("fortran_kind"));
    for (const Record *E : R->getValueAsListOfDefs("enumerators"))
      OS << "  integer(" << KindName
         << "), parameter :: " << E->getValueAsString("name") << " = "
         << E->getValueAsInt("value") << "\n";
    OS << "\n";
  }
  for (const Record *R : Records.getAllDerivedDefinitions("AccConst")) {
    StringRef KindName = recName(R->getValueAsDef("fortran_kind"));
    OS << "  integer(" << KindName << "), parameter :: " << recName(R) << " = "
       << R->getValueAsInt("value") << "\n";
  }

  std::map<std::string, std::vector<Func>> Groups;
  std::map<std::string, std::set<std::string>> Aliases;
  for (Func &F : collectFFuncs(Records)) {
    for (const std::string &A : F.Aliases)
      Aliases[F.Generic].insert(A);
    Groups[F.Generic].push_back(std::move(F));
  }
  for (auto &G : Groups)
    llvm::sort(G.second, [](const Func &A, const Func &B) {
      return A.Specific < B.Specific;
    });

  for (const auto &G : Groups) {
    OS << "\n  interface " << G.first << "\n";
    for (const Func &F : G.second) {
      emitFInterface(OS, F);
      OS << "\n";
    }
    OS << "  end interface\n";
    auto AIt = Aliases.find(G.first);
    if (AIt == Aliases.end())
      continue;
    for (const std::string &Alias : AIt->second) {
      OS << "\n  interface " << Alias << "\n";
      OS << "    procedure :: " << G.second.front().Specific;
      for (size_t I = 1; I < G.second.size(); ++I)
        OS << ", " << G.second[I].Specific;
      OS << "\n  end interface\n";
    }
  }
}

void EmitAccCEnums(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader << "\n";
  for (const Record *R : Records.getAllDerivedDefinitions("AccConst"))
    OS << "#define " << recName(R) << " (" << R->getValueAsInt("value")
       << ")\n";
  for (const Record *R : Records.getAllDerivedDefinitions("AccEnum")) {
    StringRef CName = R->getValueAsString("c_name");
    OS << "\ntypedef enum " << CName << " {\n";
    std::vector<const Record *> Etors = R->getValueAsListOfDefs("enumerators");
    for (size_t I = 0; I < Etors.size(); ++I)
      OS << "  " << Etors[I]->getValueAsString("name") << " = "
         << Etors[I]->getValueAsInt("value")
         << (I + 1 < Etors.size() ? ",\n" : "\n");
    OS << "} " << CName << ";\n";
  }
}

void EmitAccPrintEnums(const RecordKeeper &Records, raw_ostream &OS) {
  OS << GenericHeader << "\n";
  // For every enum: enumName(V) is the spelling of an enumerator, or nullptr if
  // V is not one, and enumTypeName(V) is the name of the enum type. What to
  // print for an unknown value is up to the caller.
  for (const Record *R : Records.getAllDerivedDefinitions("AccEnum")) {
    StringRef CName = R->getValueAsString("c_name");
    OS << "[[maybe_unused]] static const char *enumName(" << CName << " V) {\n";
    OS << TAB_1 << "switch (V) {\n";
    for (const Record *E : R->getValueAsListOfDefs("enumerators"))
      OS << TAB_1 << "case " << E->getValueAsString("name") << ":\n"
         << TAB_2 << "return \"" << E->getValueAsString("name") << "\";\n";
    OS << TAB_1 << "}\n";
    OS << TAB_1 << "return nullptr;\n}\n\n";
    OS << "[[maybe_unused]] static const char *enumTypeName(" << CName
       << ") {\n";
    OS << TAB_1 << "return \"" << CName << "\";\n}\n\n";
  }
  // The special async values, spelled with the constants of openacc.h.
  OS << "[[maybe_unused]] static const char *asyncConstName(int64_t V) {\n";
  OS << TAB_1 << "switch (V) {\n";
  for (const Record *R : Records.getAllDerivedDefinitions("AccConst"))
    OS << TAB_1 << "case " << recName(R) << ":\n"
       << TAB_2 << "return \"" << recName(R) << "\";\n";
  OS << TAB_1 << "}\n";
  OS << TAB_1 << "return nullptr;\n}\n\n";
}

void EmitAccExports(const RecordKeeper &Records, raw_ostream &OS) {
  OS << "VERS1.0 {\n";
  OS << "  global:\n";
  for (const Func &F : collectFFuncs(Records))
    OS << "    " << F.Bind << ";\n";
  for (const Record *R : Records.getAllDerivedDefinitions("AccExport"))
    OS << "    " << recName(R) << ";\n";
  OS << "  local:\n";
  OS << "    *;\n";
  OS << "};\n";
}
