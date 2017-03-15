//===- YamlSymbolDumper.cpp ----------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "YamlSymbolDumper.h"
#include "PdbYaml.h"
#include "YamlTypeDumper.h"

#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbackPipeline.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::codeview::yaml;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(TypeIndex)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(uint64_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(OneMethodRecord)
LLVM_YAML_IS_SEQUENCE_VECTOR(VFTableSlotKind)
LLVM_YAML_IS_SEQUENCE_VECTOR(StringRef)
LLVM_YAML_IS_SEQUENCE_VECTOR(CVType)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::pdb::yaml::PdbTpiFieldListRecord)

namespace llvm {
namespace yaml {
void ScalarEnumerationTraits<SymbolKind>::enumeration(IO &io,
                                                      SymbolKind &Value) {
  auto SymbolNames = getSymbolTypeNames();
  for (const auto &E : SymbolNames)
    io.enumCase(Value, E.Name.str().c_str(), E.Value);
}

template <> struct ScalarBitSetTraits<CompileSym2Flags> {
  static void bitset(IO &io, CompileSym2Flags &Flags) {
    auto FlagNames = getCompileSym2FlagNames();
    for (const auto &E : FlagNames) {
      io.bitSetCase(Flags, E.Name.str().c_str(),
                    static_cast<CompileSym2Flags>(E.Value));
    }
  }
};

template <> struct ScalarBitSetTraits<CompileSym3Flags> {
  static void bitset(IO &io, CompileSym3Flags &Flags) {
    auto FlagNames = getCompileSym3FlagNames();
    for (const auto &E : FlagNames) {
      io.bitSetCase(Flags, E.Name.str().c_str(),
                    static_cast<CompileSym3Flags>(E.Value));
    }
  }
};

template <> struct ScalarBitSetTraits<ExportFlags> {
  static void bitset(IO &io, ExportFlags &Flags) {
    auto FlagNames = getExportSymFlagNames();
    for (const auto &E : FlagNames) {
      io.bitSetCase(Flags, E.Name.str().c_str(),
                    static_cast<ExportFlags>(E.Value));
    }
  }
};

template <> struct ScalarBitSetTraits<LocalSymFlags> {
  static void bitset(IO &io, LocalSymFlags &Flags) {
    auto FlagNames = getLocalFlagNames();
    for (const auto &E : FlagNames) {
      io.bitSetCase(Flags, E.Name.str().c_str(),
                    static_cast<LocalSymFlags>(E.Value));
    }
  }
};

template <> struct ScalarBitSetTraits<ProcSymFlags> {
  static void bitset(IO &io, ProcSymFlags &Flags) {
    auto FlagNames = getProcSymFlagNames();
    for (const auto &E : FlagNames) {
      io.bitSetCase(Flags, E.Name.str().c_str(),
                    static_cast<ProcSymFlags>(E.Value));
    }
  }
};

template <> struct ScalarBitSetTraits<FrameProcedureOptions> {
  static void bitset(IO &io, FrameProcedureOptions &Flags) {
    auto FlagNames = getFrameProcSymFlagNames();
    for (const auto &E : FlagNames) {
      io.bitSetCase(Flags, E.Name.str().c_str(),
                    static_cast<FrameProcedureOptions>(E.Value));
    }
  }
};

template <> struct ScalarEnumerationTraits<CPUType> {
  static void enumeration(IO &io, CPUType &Cpu) {
    auto CpuNames = getCPUTypeNames();
    for (const auto &E : CpuNames) {
      io.enumCase(Cpu, E.Name.str().c_str(), static_cast<CPUType>(E.Value));
    }
  }
};

template <> struct ScalarEnumerationTraits<RegisterId> {
  static void enumeration(IO &io, RegisterId &Reg) {
    auto RegNames = getRegisterNames();
    for (const auto &E : RegNames) {
      io.enumCase(Reg, E.Name.str().c_str(), static_cast<RegisterId>(E.Value));
    }
    io.enumFallback<Hex16>(Reg);
  }
};

template <> struct ScalarEnumerationTraits<TrampolineType> {
  static void enumeration(IO &io, TrampolineType &Tramp) {
    auto TrampNames = getTrampolineNames();
    for (const auto &E : TrampNames) {
      io.enumCase(Tramp, E.Name.str().c_str(),
                  static_cast<TrampolineType>(E.Value));
    }
  }
};

template <> struct ScalarEnumerationTraits<ThunkOrdinal> {
  static void enumeration(IO &io, ThunkOrdinal &Ord) {
    auto ThunkNames = getThunkOrdinalNames();
    for (const auto &E : ThunkNames) {
      io.enumCase(Ord, E.Name.str().c_str(),
                  static_cast<ThunkOrdinal>(E.Value));
    }
  }
};

void MappingTraits<ScopeEndSym>::mapping(IO &IO, ScopeEndSym &Obj) {}

void MappingTraits<Thunk32Sym>::mapping(IO &IO, Thunk32Sym &Thunk) {
  IO.mapRequired("Parent", Thunk.Parent);
  IO.mapRequired("End", Thunk.End);
  IO.mapRequired("Next", Thunk.Next);
  IO.mapRequired("Off", Thunk.Offset);
  IO.mapRequired("Seg", Thunk.Segment);
  IO.mapRequired("Len", Thunk.Length);
  IO.mapRequired("Ordinal", Thunk.Thunk);
}

void MappingTraits<TrampolineSym>::mapping(IO &IO, TrampolineSym &Tramp) {
  IO.mapRequired("Type", Tramp.Type);
  IO.mapRequired("Size", Tramp.Size);
  IO.mapRequired("ThunkOff", Tramp.ThunkOffset);
  IO.mapRequired("TargetOff", Tramp.TargetOffset);
  IO.mapRequired("ThunkSection", Tramp.ThunkSection);
  IO.mapRequired("TargetSection", Tramp.TargetSection);
}

void MappingTraits<SectionSym>::mapping(IO &IO, SectionSym &Section) {
  IO.mapRequired("SectionNumber", Section.SectionNumber);
  IO.mapRequired("Alignment", Section.Alignment);
  IO.mapRequired("Rva", Section.Rva);
  IO.mapRequired("Length", Section.Length);
  IO.mapRequired("Characteristics", Section.Characteristics);
  IO.mapRequired("Name", Section.Name);
}

void MappingTraits<CoffGroupSym>::mapping(IO &IO, CoffGroupSym &CoffGroup) {
  IO.mapRequired("Size", CoffGroup.Size);
  IO.mapRequired("Characteristics", CoffGroup.Characteristics);
  IO.mapRequired("Offset", CoffGroup.Offset);
  IO.mapRequired("Segment", CoffGroup.Segment);
  IO.mapRequired("Name", CoffGroup.Name);
}

void MappingTraits<ExportSym>::mapping(IO &IO, ExportSym &Export) {
  IO.mapRequired("Ordinal", Export.Ordinal);
  IO.mapRequired("Flags", Export.Flags);
  IO.mapRequired("Name", Export.Name);
}

void MappingTraits<ProcSym>::mapping(IO &IO, ProcSym &Proc) {
  // TODO: Print the linkage name

  IO.mapRequired("PtrParent", Proc.Parent);
  IO.mapRequired("PtrEnd", Proc.End);
  IO.mapRequired("PtrNext", Proc.Next);
  IO.mapRequired("CodeSize", Proc.CodeSize);
  IO.mapRequired("DbgStart", Proc.DbgStart);
  IO.mapRequired("DbgEnd", Proc.DbgEnd);
  IO.mapRequired("FunctionType", Proc.FunctionType);
  IO.mapRequired("Segment", Proc.Segment);
  IO.mapRequired("Flags", Proc.Flags);
  IO.mapRequired("DisplayName", Proc.Name);
}

void MappingTraits<RegisterSym>::mapping(IO &IO, RegisterSym &Register) {
  IO.mapRequired("Type", Register.Index);
  IO.mapRequired("Seg", Register.Register);
  IO.mapRequired("Name", Register.Name);
}

void MappingTraits<PublicSym32>::mapping(IO &IO, PublicSym32 &Public) {
  IO.mapRequired("Type", Public.Index);
  IO.mapRequired("Seg", Public.Segment);
  IO.mapRequired("Off", Public.Offset);
  IO.mapRequired("Name", Public.Name);
}

void MappingTraits<ProcRefSym>::mapping(IO &IO, ProcRefSym &ProcRef) {
  IO.mapRequired("SumName", ProcRef.SumName);
  IO.mapRequired("SymOffset", ProcRef.SymOffset);
  IO.mapRequired("Mod", ProcRef.Module);
  IO.mapRequired("Name", ProcRef.Name);
}

void MappingTraits<EnvBlockSym>::mapping(IO &IO, EnvBlockSym &EnvBlock) {
  IO.mapRequired("Entries", EnvBlock.Fields);
}

void MappingTraits<InlineSiteSym>::mapping(IO &IO, InlineSiteSym &InlineSite) {
  IO.mapRequired("PtrParent", InlineSite.Parent);
  IO.mapRequired("PtrEnd", InlineSite.End);
  IO.mapRequired("Inlinee", InlineSite.Inlinee);
  // TODO: The binary annotations
}

void MappingTraits<LocalSym>::mapping(IO &IO, LocalSym &Local) {
  IO.mapRequired("Type", Local.Type);
  IO.mapRequired("Flags", Local.Flags);
  IO.mapRequired("VarName", Local.Name);
}

void MappingTraits<DefRangeSym>::mapping(IO &IO, DefRangeSym &Obj) {
  // TODO: Print the subfields
}

void MappingTraits<DefRangeSubfieldSym>::mapping(IO &IO,
                                                 DefRangeSubfieldSym &Obj) {
  // TODO: Print the subfields
}

void MappingTraits<DefRangeRegisterSym>::mapping(IO &IO,
                                                 DefRangeRegisterSym &Obj) {
  // TODO: Print the subfields
}

void MappingTraits<DefRangeFramePointerRelSym>::mapping(
    IO &IO, DefRangeFramePointerRelSym &Obj) {
  // TODO: Print the subfields
}

void MappingTraits<DefRangeSubfieldRegisterSym>::mapping(
    IO &IO, DefRangeSubfieldRegisterSym &Obj) {
  // TODO: Print the subfields
}

void MappingTraits<DefRangeFramePointerRelFullScopeSym>::mapping(
    IO &IO, DefRangeFramePointerRelFullScopeSym &Obj) {
  // TODO: Print the subfields
}

void MappingTraits<DefRangeRegisterRelSym>::mapping(
    IO &IO, DefRangeRegisterRelSym &Obj) {
  // TODO: Print the subfields
}

void MappingTraits<BlockSym>::mapping(IO &IO, BlockSym &Block) {
  // TODO: Print the linkage name
  IO.mapRequired("PtrParent", Block.Parent);
  IO.mapRequired("PtrEnd", Block.End);
  IO.mapRequired("CodeSize", Block.CodeSize);
  IO.mapRequired("Segment", Block.Segment);
  IO.mapRequired("BlockName", Block.Name);
}

void MappingTraits<LabelSym>::mapping(IO &IO, LabelSym &Label) {
  // TODO: Print the linkage name
  IO.mapRequired("Segment", Label.Segment);
  IO.mapRequired("Flags", Label.Flags);
  IO.mapRequired("Flags", Label.Flags);
  IO.mapRequired("DisplayName", Label.Name);
}

void MappingTraits<ObjNameSym>::mapping(IO &IO, ObjNameSym &ObjName) {
  IO.mapRequired("Signature", ObjName.Signature);
  IO.mapRequired("ObjectName", ObjName.Name);
}

void MappingTraits<Compile2Sym>::mapping(IO &IO, Compile2Sym &Compile2) {
  IO.mapRequired("Flags", Compile2.Flags);
  IO.mapRequired("Machine", Compile2.Machine);
  IO.mapRequired("FrontendMajor", Compile2.VersionFrontendMajor);
  IO.mapRequired("FrontendMinor", Compile2.VersionFrontendMinor);
  IO.mapRequired("FrontendBuild", Compile2.VersionFrontendBuild);
  IO.mapRequired("BackendMajor", Compile2.VersionBackendMajor);
  IO.mapRequired("BackendMinor", Compile2.VersionBackendMinor);
  IO.mapRequired("BackendBuild", Compile2.VersionBackendBuild);
  IO.mapRequired("Version", Compile2.Version);
}

void MappingTraits<Compile3Sym>::mapping(IO &IO, Compile3Sym &Compile3) {
  IO.mapRequired("Flags", Compile3.Flags);
  IO.mapRequired("Machine", Compile3.Machine);
  IO.mapRequired("FrontendMajor", Compile3.VersionFrontendMajor);
  IO.mapRequired("FrontendMinor", Compile3.VersionFrontendMinor);
  IO.mapRequired("FrontendBuild", Compile3.VersionFrontendBuild);
  IO.mapRequired("FrontendQFE", Compile3.VersionFrontendQFE);
  IO.mapRequired("BackendMajor", Compile3.VersionBackendMajor);
  IO.mapRequired("BackendMinor", Compile3.VersionBackendMinor);
  IO.mapRequired("BackendBuild", Compile3.VersionBackendBuild);
  IO.mapRequired("BackendQFE", Compile3.VersionBackendQFE);
  IO.mapRequired("Version", Compile3.Version);
}

void MappingTraits<FrameProcSym>::mapping(IO &IO, FrameProcSym &FrameProc) {
  IO.mapRequired("TotalFrameBytes", FrameProc.TotalFrameBytes);
  IO.mapRequired("PaddingFrameBytes", FrameProc.PaddingFrameBytes);
  IO.mapRequired("OffsetToPadding", FrameProc.OffsetToPadding);
  IO.mapRequired("BytesOfCalleeSavedRegisters",
                 FrameProc.BytesOfCalleeSavedRegisters);
  IO.mapRequired("OffsetOfExceptionHandler",
                 FrameProc.OffsetOfExceptionHandler);
  IO.mapRequired("SectionIdOfExceptionHandler",
                 FrameProc.SectionIdOfExceptionHandler);
  IO.mapRequired("Flags", FrameProc.Flags);
}

void MappingTraits<CallSiteInfoSym>::mapping(IO &IO,
                                             CallSiteInfoSym &CallSiteInfo) {
  // TODO: Map Linkage Name
  IO.mapRequired("Segment", CallSiteInfo.Segment);
  IO.mapRequired("Type", CallSiteInfo.Type);
}

void MappingTraits<FileStaticSym>::mapping(IO &IO, FileStaticSym &FileStatic) {
  IO.mapRequired("Index", FileStatic.Index);
  IO.mapRequired("ModFilenameOffset", FileStatic.ModFilenameOffset);
  IO.mapRequired("Flags", FileStatic.Flags);
  IO.mapRequired("Name", FileStatic.Name);
}

void MappingTraits<HeapAllocationSiteSym>::mapping(
    IO &IO, HeapAllocationSiteSym &HeapAllocSite) {
  // TODO: Map Linkage Name
  IO.mapRequired("Segment", HeapAllocSite.Segment);
  IO.mapRequired("CallInstructionSize", HeapAllocSite.CallInstructionSize);
  IO.mapRequired("Type", HeapAllocSite.Type);
}

void MappingTraits<FrameCookieSym>::mapping(IO &IO,
                                            FrameCookieSym &FrameCookie) {
  // TODO: Map Linkage Name
  IO.mapRequired("Register", FrameCookie.Register);
  IO.mapRequired("CookieKind", FrameCookie.CookieKind);
  IO.mapRequired("Flags", FrameCookie.Flags);
}

void MappingTraits<CallerSym>::mapping(IO &IO, CallerSym &Caller) {
  // TODO: Correctly handle the ArrayRef in here.
  std::vector<TypeIndex> Indices(Caller.Indices);
  IO.mapRequired("FuncID", Indices);
}

void MappingTraits<UDTSym>::mapping(IO &IO, UDTSym &UDT) {
  IO.mapRequired("Type", UDT.Type);
  IO.mapRequired("UDTName", UDT.Name);
}

void MappingTraits<BuildInfoSym>::mapping(IO &IO, BuildInfoSym &BuildInfo) {
  IO.mapRequired("BuildId", BuildInfo.BuildId);
}

void MappingTraits<BPRelativeSym>::mapping(IO &IO, BPRelativeSym &BPRel) {
  IO.mapRequired("Offset", BPRel.Offset);
  IO.mapRequired("Type", BPRel.Type);
  IO.mapRequired("VarName", BPRel.Name);
}

void MappingTraits<RegRelativeSym>::mapping(IO &IO, RegRelativeSym &RegRel) {
  IO.mapRequired("Offset", RegRel.Offset);
  IO.mapRequired("Type", RegRel.Type);
  IO.mapRequired("Register", RegRel.Register);
  IO.mapRequired("VarName", RegRel.Name);
}

void MappingTraits<ConstantSym>::mapping(IO &IO, ConstantSym &Constant) {
  IO.mapRequired("Type", Constant.Type);
  IO.mapRequired("Value", Constant.Value);
  IO.mapRequired("Name", Constant.Name);
}

void MappingTraits<DataSym>::mapping(IO &IO, DataSym &Data) {
  // TODO: Map linkage name
  IO.mapRequired("Type", Data.Type);
  IO.mapRequired("DisplayName", Data.Name);
}

void MappingTraits<ThreadLocalDataSym>::mapping(IO &IO,
                                                ThreadLocalDataSym &Data) {
  // TODO: Map linkage name
  IO.mapRequired("Type", Data.Type);
  IO.mapRequired("DisplayName", Data.Name);
}
}
}

Error llvm::codeview::yaml::YamlSymbolDumper::visitSymbolBegin(CVSymbol &CVR) {
  YamlIO.mapRequired("Kind", CVR.Type);
  return Error::success();
}
