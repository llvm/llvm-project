//===--- ReadObjAnalyzer.cpp - LLVM Advisor ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Binary/ReadObjAnalyzer.h"
#include "Analysis/Binary/BinaryAnalysisUtils.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
ReadObjAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  return withObjectFile(Context, CapID, UnitID, [&](const object::ObjectFile &Obj) {
    json::Array Sections;
    for (object::SectionRef Sec : Obj.sections()) {
      json::Object S;
      if (Expected<StringRef> Name = Sec.getName())
        S["name"] = *Name;
      else
        S["name"] = "";
      S["size"] = static_cast<int64_t>(Sec.getSize());
      Sections.push_back(std::move(S));
    }

    json::Array Symbols;
    for (object::SymbolRef Sym : Obj.symbols()) {
      if (Expected<StringRef> Name = Sym.getName()) {
        json::Object S;
        S["name"] = *Name;
        if (Expected<object::SymbolRef::Type> Ty = Sym.getType())
          S["type"] = static_cast<int64_t>(*Ty);
        Symbols.push_back(std::move(S));
      }
    }

    return makeJSONResult(CapID, UnitID, json::Object{
        {"format", Obj.getFileFormatName()},
        {"sections", std::move(Sections)},
        {"symbol_count", static_cast<int64_t>(Symbols.size())}});
  });
}
