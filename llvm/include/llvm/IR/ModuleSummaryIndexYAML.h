//===-- llvm/ModuleSummaryIndexYAML.h - YAML I/O for summary ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_MODULESUMMARYINDEXYAML_H
#define LLVM_IR_MODULESUMMARYINDEXYAML_H

#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<TypeTestResolution::Kind> {
  static void enumeration(IO &io, TypeTestResolution::Kind &value) {
    io.enumCase(value, "Unsat", TypeTestResolution::Unsat);
    io.enumCase(value, "ByteArray", TypeTestResolution::ByteArray);
    io.enumCase(value, "Inline", TypeTestResolution::Inline);
    io.enumCase(value, "Single", TypeTestResolution::Single);
    io.enumCase(value, "AllOnes", TypeTestResolution::AllOnes);
  }
};

template <> struct MappingTraits<TypeTestResolution> {
  static void mapping(IO &io, TypeTestResolution &res) {
    io.mapOptional("Kind", res.TheKind);
    io.mapOptional("SizeM1BitWidth", res.SizeM1BitWidth);
  }
};

template <> struct MappingTraits<TypeIdSummary> {
  static void mapping(IO &io, TypeIdSummary& summary) {
    io.mapOptional("TTRes", summary.TTRes);
  }
};

struct FunctionSummaryYaml {
  std::vector<uint64_t> TypeTests;
};

} // End yaml namespace
} // End llvm namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(uint64_t)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<FunctionSummaryYaml> {
  static void mapping(IO &io, FunctionSummaryYaml& summary) {
    io.mapOptional("TypeTests", summary.TypeTests);
  }
};

} // End yaml namespace
} // End llvm namespace

LLVM_YAML_IS_STRING_MAP(TypeIdSummary)
LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionSummaryYaml)

namespace llvm {
namespace yaml {

// FIXME: Add YAML mappings for the rest of the module summary.
template <> struct CustomMappingTraits<GlobalValueSummaryMapTy> {
  static void inputOne(IO &io, StringRef Key, GlobalValueSummaryMapTy &V) {
    std::vector<FunctionSummaryYaml> FSums;
    io.mapRequired(Key.str().c_str(), FSums);
    uint64_t KeyInt;
    if (Key.getAsInteger(0, KeyInt)) {
      io.setError("key not an integer");
      return;
    }
    auto &Elem = V[KeyInt];
    for (auto &FSum : FSums) {
      GlobalValueSummary::GVFlags GVFlags(GlobalValue::ExternalLinkage, false,
                                          false);
      Elem.push_back(llvm::make_unique<FunctionSummary>(
          GVFlags, 0, ArrayRef<ValueInfo>{},
          ArrayRef<FunctionSummary::EdgeTy>{}, std::move(FSum.TypeTests)));
    }
  }
  static void output(IO &io, GlobalValueSummaryMapTy &V) {
    for (auto &P : V) {
      std::vector<FunctionSummaryYaml> FSums;
      for (auto &Sum : P.second) {
        if (auto *FSum = dyn_cast<FunctionSummary>(Sum.get()))
          FSums.push_back(FunctionSummaryYaml{FSum->type_tests()});
      }
      if (!FSums.empty())
        io.mapRequired(llvm::utostr(P.first).c_str(), FSums);
    }
  }
};

template <> struct MappingTraits<ModuleSummaryIndex> {
  static void mapping(IO &io, ModuleSummaryIndex& index) {
    io.mapOptional("GlobalValueMap", index.GlobalValueMap);
    io.mapOptional("TypeIdMap", index.TypeIdMap);
  }
};

} // End yaml namespace
} // End llvm namespace

#endif
