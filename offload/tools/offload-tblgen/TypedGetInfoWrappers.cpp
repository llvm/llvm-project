//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a Tablegen backend that produces typed C++ inline wrappers for
// various `olGet*Info interfaces.
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

void EmitTypedGetInfoWrappers(const llvm::RecordKeeper &Records,
                              llvm::raw_ostream &OS) {
  OS << GenericHeader;
  for (auto *R : Records.getAllDerivedDefinitions("Function")) {
    auto Name = R->getName();
    if (!Name.starts_with("olGet") || !Name.ends_with("Info"))
      continue;
    auto F = FunctionRec{R};
    auto Params = F.getParams();
    assert(Params.size() == 4);
    auto Object = Params[0];
    auto InfoDesc = Params[1];

    OS << formatv("template <{} Desc> inline auto get_info({} {});\n",
                  InfoDesc.getType(), Object.getType(), Object.getName());

    EnumRec E{Records.getDef(InfoDesc.getType())};
    for (auto &V : E.getValues()) {
      auto Desc = E.getEnumValNamePrefix() + "_" + V.getName();
      auto TaggedType = V.getTaggedType();
      auto ElementType = TaggedType.rtrim("[]");
      auto ResultType = [&]() -> std::string {
        if (!TaggedType.ends_with("[]"))
          return TaggedType.str();
        if (TaggedType == "char[]")
          return "std::string";

        return ("std::vector<" + ElementType + ">").str();
      }();
      auto ReturnType =
          "std::variant<" + ResultType + ", " + PrefixLower + "_result_t>";
      OS << formatv("template<> inline auto get_info<{}>({} {}) {{\n", Desc,
                    Object.getType(), Object.getName());
      if (TaggedType.ends_with("[]")) {
        OS << TAB_1 << formatv("{0} Result;\n", ResultType);
        OS << TAB_1 << "size_t ResultSize = 0;\n";
        OS << TAB_1
           << formatv("if (auto Err = {}Size({}, {}, &ResultSize))\n",
                      F.getName(), Object.getName(), Desc);
        OS << TAB_2 << formatv("return {}{{Err};\n", ReturnType);
        if (TaggedType == "char[]") {
          // Null terminator isn't counted in std::string::size.
          OS << TAB_1 << "ResultSize -= 1;\n";
        } else {
          OS << TAB_1
             << formatv("assert(ResultSize % sizeof({}) == 0);\n", ElementType);
          OS << TAB_1 << formatv("ResultSize /= sizeof({});\n", ElementType);
        }
        OS << TAB_1 << "Result.resize(ResultSize);\n";
        OS << TAB_1
           << formatv("if (auto Err = {}({}, {}, ResultSize, Result.data()))\n",
                      F.getName(), Object.getName(), Desc);
        OS << TAB_2 << formatv("return {0}{{Err};\n", ReturnType);
        OS << TAB_1 << "else\n";
        OS << TAB_2 << formatv("return {0}{{Result};\n", ReturnType);
      } else {
        OS << TAB_1 << formatv("{0} Result;\n", TaggedType);
        OS << TAB_1
           << formatv("if (auto Err = {}({}, {}, 1, &Result))\n", F.getName(),
                      Object.getName(), Desc);
        OS << TAB_2 << formatv("return {0}{{Err};\n", ReturnType);
        OS << TAB_1 << "else\n";
        OS << TAB_2 << formatv("return {0}{{Result};\n", ReturnType);
      }
      OS << "}\n";
    }
    OS << "\n";
  }
}
