//===-- llvm/ModuleSummaryIndexYAML.h - YAML I/O for summary ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    io.enumCase(value, "Unknown", TypeTestResolution::Unknown);
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
    io.mapOptional("AlignLog2", res.AlignLog2);
    io.mapOptional("SizeM1", res.SizeM1);
    io.mapOptional("BitMask", res.BitMask);
    io.mapOptional("InlineBits", res.InlineBits);
  }
};

template <>
struct ScalarEnumerationTraits<WholeProgramDevirtResolution::ByArg::Kind> {
  static void enumeration(IO &io,
                          WholeProgramDevirtResolution::ByArg::Kind &value) {
    io.enumCase(value, "Indir", WholeProgramDevirtResolution::ByArg::Indir);
    io.enumCase(value, "UniformRetVal",
                WholeProgramDevirtResolution::ByArg::UniformRetVal);
    io.enumCase(value, "UniqueRetVal",
                WholeProgramDevirtResolution::ByArg::UniqueRetVal);
    io.enumCase(value, "VirtualConstProp",
                WholeProgramDevirtResolution::ByArg::VirtualConstProp);
  }
};

template <> struct MappingTraits<WholeProgramDevirtResolution::ByArg> {
  static void mapping(IO &io, WholeProgramDevirtResolution::ByArg &res) {
    io.mapOptional("Kind", res.TheKind);
    io.mapOptional("Info", res.Info);
    io.mapOptional("Byte", res.Byte);
    io.mapOptional("Bit", res.Bit);
  }
};

template <>
struct CustomMappingTraits<
    std::map<std::vector<uint64_t>, WholeProgramDevirtResolution::ByArg>> {
  static void inputOne(
      IO &io, StringRef Key,
      std::map<std::vector<uint64_t>, WholeProgramDevirtResolution::ByArg> &V) {
    std::vector<uint64_t> Args;
    std::pair<StringRef, StringRef> P = {"", Key};
    while (!P.second.empty()) {
      P = P.second.split(',');
      uint64_t Arg;
      if (P.first.getAsInteger(0, Arg)) {
        io.setError("key not an integer");
        return;
      }
      Args.push_back(Arg);
    }
    io.mapRequired(Key.str().c_str(), V[Args]);
  }
  static void output(
      IO &io,
      std::map<std::vector<uint64_t>, WholeProgramDevirtResolution::ByArg> &V) {
    for (auto &P : V) {
      std::string Key;
      for (uint64_t Arg : P.first) {
        if (!Key.empty())
          Key += ',';
        Key += llvm::utostr(Arg);
      }
      io.mapRequired(Key.c_str(), P.second);
    }
  }
};

template <> struct ScalarEnumerationTraits<WholeProgramDevirtResolution::Kind> {
  static void enumeration(IO &io, WholeProgramDevirtResolution::Kind &value) {
    io.enumCase(value, "Indir", WholeProgramDevirtResolution::Indir);
    io.enumCase(value, "SingleImpl", WholeProgramDevirtResolution::SingleImpl);
    io.enumCase(value, "BranchFunnel",
                WholeProgramDevirtResolution::BranchFunnel);
  }
};

template <> struct MappingTraits<WholeProgramDevirtResolution> {
  static void mapping(IO &io, WholeProgramDevirtResolution &res) {
    io.mapOptional("Kind", res.TheKind);
    io.mapOptional("SingleImplName", res.SingleImplName);
    io.mapOptional("ResByArg", res.ResByArg);
  }
};

template <>
struct CustomMappingTraits<std::map<uint64_t, WholeProgramDevirtResolution>> {
  static void inputOne(IO &io, StringRef Key,
                       std::map<uint64_t, WholeProgramDevirtResolution> &V) {
    uint64_t KeyInt;
    if (Key.getAsInteger(0, KeyInt)) {
      io.setError("key not an integer");
      return;
    }
    io.mapRequired(Key.str().c_str(), V[KeyInt]);
  }
  static void output(IO &io, std::map<uint64_t, WholeProgramDevirtResolution> &V) {
    for (auto &P : V)
      io.mapRequired(llvm::utostr(P.first).c_str(), P.second);
  }
};

template <> struct MappingTraits<TypeIdSummary> {
  static void mapping(IO &io, TypeIdSummary& summary) {
    io.mapOptional("TTRes", summary.TTRes);
    io.mapOptional("WPDRes", summary.WPDRes);
  }
};

struct GlobalValueSummaryYaml {
  // Commonly used fields
  unsigned Linkage, Visibility;
  bool NotEligibleToImport, Live, IsLocal, CanAutoHide;
  unsigned ImportType;
  // Fields for AliasSummary
  std::optional<uint64_t> Aliasee;
  // Fields for FunctionSummary
  std::vector<uint64_t> Refs = {};
  std::vector<uint64_t> TypeTests = {};
  std::vector<FunctionSummary::VFuncId> TypeTestAssumeVCalls = {};
  std::vector<FunctionSummary::VFuncId> TypeCheckedLoadVCalls = {};
  std::vector<FunctionSummary::ConstVCall> TypeTestAssumeConstVCalls = {};
  std::vector<FunctionSummary::ConstVCall> TypeCheckedLoadConstVCalls = {};
};

} // End yaml namespace
} // End llvm namespace

namespace llvm {
namespace yaml {

template <> struct MappingTraits<FunctionSummary::VFuncId> {
  static void mapping(IO &io, FunctionSummary::VFuncId& id) {
    io.mapOptional("GUID", id.GUID);
    io.mapOptional("Offset", id.Offset);
  }
};

template <> struct MappingTraits<FunctionSummary::ConstVCall> {
  static void mapping(IO &io, FunctionSummary::ConstVCall& id) {
    io.mapOptional("VFunc", id.VFunc);
    io.mapOptional("Args", id.Args);
  }
};

} // End yaml namespace
} // End llvm namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionSummary::VFuncId)
LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionSummary::ConstVCall)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<GlobalValueSummaryYaml> {
  static void mapping(IO &io, GlobalValueSummaryYaml &summary) {
    io.mapOptional("Linkage", summary.Linkage);
    io.mapOptional("Visibility", summary.Visibility);
    io.mapOptional("NotEligibleToImport", summary.NotEligibleToImport);
    io.mapOptional("Live", summary.Live);
    io.mapOptional("Local", summary.IsLocal);
    io.mapOptional("CanAutoHide", summary.CanAutoHide);
    io.mapOptional("ImportType", summary.ImportType);
    io.mapOptional("Aliasee", summary.Aliasee);
    io.mapOptional("Refs", summary.Refs);
    io.mapOptional("TypeTests", summary.TypeTests);
    io.mapOptional("TypeTestAssumeVCalls", summary.TypeTestAssumeVCalls);
    io.mapOptional("TypeCheckedLoadVCalls", summary.TypeCheckedLoadVCalls);
    io.mapOptional("TypeTestAssumeConstVCalls",
                   summary.TypeTestAssumeConstVCalls);
    io.mapOptional("TypeCheckedLoadConstVCalls",
                   summary.TypeCheckedLoadConstVCalls);
  }
};

} // End yaml namespace
} // End llvm namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(GlobalValueSummaryYaml)

namespace llvm {
namespace yaml {

// FIXME: Add YAML mappings for the rest of the module summary.
template <> struct CustomMappingTraits<GlobalValueSummaryMapTy> {
  static void inputOne(IO &io, StringRef Key, GlobalValueSummaryMapTy &V) {
    std::vector<GlobalValueSummaryYaml> GVSums;
    io.mapRequired(Key.str().c_str(), GVSums);
    uint64_t KeyInt;
    if (Key.getAsInteger(0, KeyInt)) {
      io.setError("key not an integer");
      return;
    }
    auto &Elem = V.try_emplace(KeyInt, /*IsAnalysis=*/false).first->second;
    for (auto &GVSum : GVSums) {
      GlobalValueSummary::GVFlags GVFlags(
          static_cast<GlobalValue::LinkageTypes>(GVSum.Linkage),
          static_cast<GlobalValue::VisibilityTypes>(GVSum.Visibility),
          GVSum.NotEligibleToImport, GVSum.Live, GVSum.IsLocal,
          GVSum.CanAutoHide,
          static_cast<GlobalValueSummary::ImportKind>(GVSum.ImportType));
      if (GVSum.Aliasee) {
        auto ASum = std::make_unique<AliasSummary>(GVFlags);
        if (!V.count(*GVSum.Aliasee))
          V.emplace(*GVSum.Aliasee, /*IsAnalysis=*/false);
        ValueInfo AliaseeVI(/*IsAnalysis=*/false, &*V.find(*GVSum.Aliasee));
        // Note: Aliasee cannot be filled until all summaries are loaded.
        // This is done in fixAliaseeLinks() which is called in
        // MappingTraits<ModuleSummaryIndex>::mapping().
        ASum->setAliasee(AliaseeVI, /*Aliasee=*/nullptr);
        Elem.SummaryList.push_back(std::move(ASum));
        continue;
      }
      SmallVector<ValueInfo, 0> Refs;
      Refs.reserve(GVSum.Refs.size());
      for (auto &RefGUID : GVSum.Refs) {
        auto It = V.try_emplace(RefGUID, /*IsAnalysis=*/false).first;
        Refs.push_back(ValueInfo(/*IsAnalysis=*/false, &*It));
      }
      Elem.SummaryList.push_back(std::make_unique<FunctionSummary>(
          GVFlags, /*NumInsts=*/0, FunctionSummary::FFlags{}, std::move(Refs),
          SmallVector<FunctionSummary::EdgeTy, 0>{}, std::move(GVSum.TypeTests),
          std::move(GVSum.TypeTestAssumeVCalls),
          std::move(GVSum.TypeCheckedLoadVCalls),
          std::move(GVSum.TypeTestAssumeConstVCalls),
          std::move(GVSum.TypeCheckedLoadConstVCalls),
          ArrayRef<FunctionSummary::ParamAccess>{}, ArrayRef<CallsiteInfo>{},
          ArrayRef<AllocInfo>{}));
    }
  }
  static void output(IO &io, GlobalValueSummaryMapTy &V) {
    for (auto &P : V) {
      std::vector<GlobalValueSummaryYaml> GVSums;
      for (auto &Sum : P.second.SummaryList) {
        if (auto *FSum = dyn_cast<FunctionSummary>(Sum.get())) {
          std::vector<uint64_t> Refs;
          Refs.reserve(FSum->refs().size());
          for (auto &VI : FSum->refs())
            Refs.push_back(VI.getGUID());
          GVSums.push_back(GlobalValueSummaryYaml{
              FSum->flags().Linkage, FSum->flags().Visibility,
              static_cast<bool>(FSum->flags().NotEligibleToImport),
              static_cast<bool>(FSum->flags().Live),
              static_cast<bool>(FSum->flags().DSOLocal),
              static_cast<bool>(FSum->flags().CanAutoHide),
              FSum->flags().ImportType, /*Aliasee=*/std::nullopt, Refs,
              FSum->type_tests(), FSum->type_test_assume_vcalls(),
              FSum->type_checked_load_vcalls(),
              FSum->type_test_assume_const_vcalls(),
              FSum->type_checked_load_const_vcalls()});
        } else if (auto *ASum = dyn_cast<AliasSummary>(Sum.get());
                   ASum && ASum->hasAliasee()) {
          GVSums.push_back(GlobalValueSummaryYaml{
              ASum->flags().Linkage, ASum->flags().Visibility,
              static_cast<bool>(ASum->flags().NotEligibleToImport),
              static_cast<bool>(ASum->flags().Live),
              static_cast<bool>(ASum->flags().DSOLocal),
              static_cast<bool>(ASum->flags().CanAutoHide),
              ASum->flags().ImportType,
              /*Aliasee=*/ASum->getAliaseeGUID()});
        }
      }
      if (!GVSums.empty())
        io.mapRequired(llvm::utostr(P.first).c_str(), GVSums);
    }
  }
  static void fixAliaseeLinks(GlobalValueSummaryMapTy &V) {
    for (auto &P : V) {
      for (auto &Sum : P.second.SummaryList) {
        if (auto *Alias = dyn_cast<AliasSummary>(Sum.get())) {
          ValueInfo AliaseeVI = Alias->getAliaseeVI();
          auto AliaseeSL = AliaseeVI.getSummaryList();
          if (AliaseeSL.empty()) {
            ValueInfo EmptyVI;
            Alias->setAliasee(EmptyVI, nullptr);
          } else
            Alias->setAliasee(AliaseeVI, AliaseeSL[0].get());
        }
      }
    }
  }
};

template <> struct CustomMappingTraits<TypeIdSummaryMapTy> {
  static void inputOne(IO &io, StringRef Key, TypeIdSummaryMapTy &V) {
    TypeIdSummary TId;
    io.mapRequired(Key.str().c_str(), TId);
    V.insert({GlobalValue::getGUID(Key), {Key, TId}});
  }
  static void output(IO &io, TypeIdSummaryMapTy &V) {
    for (auto &TidIter : V)
      io.mapRequired(TidIter.second.first.str().c_str(), TidIter.second.second);
  }
};

template <> struct MappingTraits<ModuleSummaryIndex> {
  static void mapping(IO &io, ModuleSummaryIndex& index) {
    io.mapOptional("GlobalValueMap", index.GlobalValueMap);
    if (!io.outputting())
      CustomMappingTraits<GlobalValueSummaryMapTy>::fixAliaseeLinks(
          index.GlobalValueMap);

    if (io.outputting()) {
      io.mapOptional("TypeIdMap", index.TypeIdMap);
    } else {
      TypeIdSummaryMapTy TypeIdMap;
      io.mapOptional("TypeIdMap", TypeIdMap);
      for (auto &[TypeGUID, TypeIdSummaryMap] : TypeIdMap) {
        // Save type id references in index and point TypeIdMap to use the
        // references owned by index.
        StringRef KeyRef = index.TypeIdSaver.save(TypeIdSummaryMap.first);
        index.TypeIdMap.insert(
            {TypeGUID, {KeyRef, std::move(TypeIdSummaryMap.second)}});
      }
    }

    io.mapOptional("WithGlobalValueDeadStripping",
                   index.WithGlobalValueDeadStripping);

    if (io.outputting()) {
      std::vector<std::string> CfiFunctionDefs(index.CfiFunctionDefs.begin(),
                                               index.CfiFunctionDefs.end());
      io.mapOptional("CfiFunctionDefs", CfiFunctionDefs);
      std::vector<std::string> CfiFunctionDecls(index.CfiFunctionDecls.begin(),
                                                index.CfiFunctionDecls.end());
      io.mapOptional("CfiFunctionDecls", CfiFunctionDecls);
    } else {
      std::vector<std::string> CfiFunctionDefs;
      io.mapOptional("CfiFunctionDefs", CfiFunctionDefs);
      index.CfiFunctionDefs = {CfiFunctionDefs.begin(), CfiFunctionDefs.end()};
      std::vector<std::string> CfiFunctionDecls;
      io.mapOptional("CfiFunctionDecls", CfiFunctionDecls);
      index.CfiFunctionDecls = {CfiFunctionDecls.begin(),
                                CfiFunctionDecls.end()};
    }
  }
};

} // End yaml namespace
} // End llvm namespace

#endif
