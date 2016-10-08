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

template <> struct ScalarEnumerationTraits<ThunkOrdinal> {
  void enumeration(IO &io, ThunkOrdinal Ord) {}
};

void MappingTraits<ScopeEndSym>::mapping(IO &IO, ScopeEndSym &Obj) {}

void MappingTraits<Thunk32Sym>::mapping(IO &IO, Thunk32Sym &Thunk) {
  IO.mapRequired("Parent", Thunk.Header.Parent);
  IO.mapRequired("End", Thunk.Header.End);
  IO.mapRequired("Next", Thunk.Header.Next);
  IO.mapRequired("Off", Thunk.Header.Off);
  IO.mapRequired("Seg", Thunk.Header.Seg);
  IO.mapRequired("Len", Thunk.Header.Len);
  IO.mapRequired("Ordinal", Thunk.Header.Ord);
}

void MappingTraits<TrampolineSym>::mapping(IO &IO, TrampolineSym &Tramp) {
  IO.mapRequired("Type", Tramp.Header.Type);
  IO.mapRequired("Size", Tramp.Header.Size);
  IO.mapRequired("ThunkOff", Tramp.Header.ThunkOff);
  IO.mapRequired("TargetOff", Tramp.Header.TargetOff);
  IO.mapRequired("ThunkSection", Tramp.Header.ThunkSection);
  IO.mapRequired("TargetSection", Tramp.Header.TargetSection);
}

void MappingTraits<SectionSym>::mapping(IO &IO, SectionSym &Section) {
  IO.mapRequired("SectionNumber", Section.Header.SectionNumber);
  IO.mapRequired("Alignment", Section.Header.Alignment);
  IO.mapRequired("Reserved", Section.Header.Reserved);
  IO.mapRequired("Rva", Section.Header.Rva);
  IO.mapRequired("Length", Section.Header.Length);
  IO.mapRequired("Characteristics", Section.Header.Characteristics);
  IO.mapRequired("Name", Section.Name);
}

void MappingTraits<CoffGroupSym>::mapping(IO &IO, CoffGroupSym &CoffGroup) {
  IO.mapRequired("Size", CoffGroup.Header.Size);
  IO.mapRequired("Characteristics", CoffGroup.Header.Characteristics);
  IO.mapRequired("Offset", CoffGroup.Header.Offset);
  IO.mapRequired("Segment", CoffGroup.Header.Segment);
  IO.mapRequired("Name", CoffGroup.Name);
}

void MappingTraits<ExportSym>::mapping(IO &IO, ExportSym &Export) {
  IO.mapRequired("Ordinal", Export.Header.Ordinal);
  IO.mapRequired("Flags", Export.Header.Flags);
  IO.mapRequired("Name", Export.Name);
}

void MappingTraits<ProcSym>::mapping(IO &IO, ProcSym &Proc) {
  // TODO: Print the linkage name

  IO.mapRequired("PtrParent", Proc.Header.PtrParent);
  IO.mapRequired("PtrEnd", Proc.Header.PtrEnd);
  IO.mapRequired("PtrNext", Proc.Header.PtrNext);
  IO.mapRequired("CodeSize", Proc.Header.CodeSize);
  IO.mapRequired("DbgStart", Proc.Header.DbgStart);
  IO.mapRequired("DbgEnd", Proc.Header.DbgEnd);
  IO.mapRequired("FunctionType", Proc.Header.FunctionType);
  IO.mapRequired("Segment", Proc.Header.Segment);
  IO.mapRequired("Flags", Proc.Header.Flags);
  IO.mapRequired("DisplayName", Proc.Name);
}

void MappingTraits<RegisterSym>::mapping(IO &IO, RegisterSym &Register) {
  IO.mapRequired("Type", Register.Header.Index);
  IO.mapRequired("Seg", Register.Header.Register);
  IO.mapRequired("Name", Register.Name);
}

void MappingTraits<PublicSym32>::mapping(IO &IO, PublicSym32 &Public) {
  IO.mapRequired("Type", Public.Header.Index);
  IO.mapRequired("Seg", Public.Header.Seg);
  IO.mapRequired("Off", Public.Header.Off);
  IO.mapRequired("Name", Public.Name);
}

void MappingTraits<ProcRefSym>::mapping(IO &IO, ProcRefSym &ProcRef) {
  IO.mapRequired("SumName", ProcRef.Header.SumName);
  IO.mapRequired("SymOffset", ProcRef.Header.SymOffset);
  IO.mapRequired("Mod", ProcRef.Header.Mod);
  IO.mapRequired("Name", ProcRef.Name);
}

void MappingTraits<EnvBlockSym>::mapping(IO &IO, EnvBlockSym &EnvBlock) {
  IO.mapRequired("Reserved", EnvBlock.Header.Reserved);
  IO.mapRequired("Entries", EnvBlock.Fields);
}

void MappingTraits<InlineSiteSym>::mapping(IO &IO, InlineSiteSym &InlineSite) {
  IO.mapRequired("PtrParent", InlineSite.Header.PtrParent);
  IO.mapRequired("PtrEnd", InlineSite.Header.PtrEnd);
  IO.mapRequired("Inlinee", InlineSite.Header.Inlinee);
  // TODO: The binary annotations
}

void MappingTraits<LocalSym>::mapping(IO &IO, LocalSym &Local) {
  IO.mapRequired("Type", Local.Header.Type);
  IO.mapRequired("Flags", Local.Header.Flags);
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
  IO.mapRequired("PtrParent", Block.Header.PtrParent);
  IO.mapRequired("PtrEnd", Block.Header.PtrEnd);
  IO.mapRequired("CodeSize", Block.Header.CodeSize);
  IO.mapRequired("Segment", Block.Header.Segment);
  IO.mapRequired("BlockName", Block.Name);
}

void MappingTraits<LabelSym>::mapping(IO &IO, LabelSym &Label) {
  // TODO: Print the linkage name
  IO.mapRequired("Segment", Label.Header.Segment);
  IO.mapRequired("Flags", Label.Header.Flags);
  IO.mapRequired("Flags", Label.Header.Flags);
  IO.mapRequired("DisplayName", Label.Name);
}

void MappingTraits<ObjNameSym>::mapping(IO &IO, ObjNameSym &ObjName) {
  IO.mapRequired("Signature", ObjName.Header.Signature);
  IO.mapRequired("ObjectName", ObjName.Name);
}

void MappingTraits<Compile2Sym>::mapping(IO &IO, Compile2Sym &Compile2) {
  IO.mapRequired("Flags", Compile2.Header.flags);
  IO.mapRequired("Machine", Compile2.Header.Machine);
  IO.mapRequired("FrontendMajor", Compile2.Header.VersionFrontendMajor);
  IO.mapRequired("FrontendMinor", Compile2.Header.VersionFrontendMinor);
  IO.mapRequired("FrontendBuild", Compile2.Header.VersionFrontendBuild);
  IO.mapRequired("BackendMajor", Compile2.Header.VersionBackendMajor);
  IO.mapRequired("BackendMinor", Compile2.Header.VersionBackendMinor);
  IO.mapRequired("BackendBuild", Compile2.Header.VersionBackendBuild);
  IO.mapRequired("Version", Compile2.Version);
}

void MappingTraits<Compile3Sym>::mapping(IO &IO, Compile3Sym &Compile3) {
  IO.mapRequired("Flags", Compile3.Header.flags);
  IO.mapRequired("Machine", Compile3.Header.Machine);
  IO.mapRequired("FrontendMajor", Compile3.Header.VersionFrontendMajor);
  IO.mapRequired("FrontendMinor", Compile3.Header.VersionFrontendMinor);
  IO.mapRequired("FrontendBuild", Compile3.Header.VersionFrontendBuild);
  IO.mapRequired("FrontendQFE", Compile3.Header.VersionFrontendQFE);
  IO.mapRequired("BackendMajor", Compile3.Header.VersionBackendMajor);
  IO.mapRequired("BackendMinor", Compile3.Header.VersionBackendMinor);
  IO.mapRequired("BackendBuild", Compile3.Header.VersionBackendBuild);
  IO.mapRequired("BackendQFE", Compile3.Header.VersionBackendQFE);
  IO.mapRequired("Version", Compile3.Version);
}

void MappingTraits<FrameProcSym>::mapping(IO &IO, FrameProcSym &FrameProc) {
  IO.mapRequired("TotalFrameBytes", FrameProc.Header.TotalFrameBytes);
  IO.mapRequired("PaddingFrameBytes", FrameProc.Header.PaddingFrameBytes);
  IO.mapRequired("OffsetToPadding", FrameProc.Header.OffsetToPadding);
  IO.mapRequired("BytesOfCalleeSavedRegisters",
                 FrameProc.Header.BytesOfCalleeSavedRegisters);
  IO.mapRequired("OffsetOfExceptionHandler",
                 FrameProc.Header.OffsetOfExceptionHandler);
  IO.mapRequired("SectionIdOfExceptionHandler",
                 FrameProc.Header.SectionIdOfExceptionHandler);
  IO.mapRequired("Flags", FrameProc.Header.Flags);
}

void MappingTraits<CallSiteInfoSym>::mapping(IO &IO,
                                             CallSiteInfoSym &CallSiteInfo) {
  // TODO: Map Linkage Name
  IO.mapRequired("Segment", CallSiteInfo.Header.Segment);
  IO.mapRequired("Reserved", CallSiteInfo.Header.Reserved);
  IO.mapRequired("Type", CallSiteInfo.Header.Type);
}

void MappingTraits<FileStaticSym>::mapping(IO &IO, FileStaticSym &FileStatic) {
  IO.mapRequired("Index", FileStatic.Header.Index);
  IO.mapRequired("ModFilenameOffset", FileStatic.Header.ModFilenameOffset);
  IO.mapRequired("Flags", FileStatic.Header.Flags);
  IO.mapRequired("Name", FileStatic.Name);
}

void MappingTraits<HeapAllocationSiteSym>::mapping(
    IO &IO, HeapAllocationSiteSym &HeapAllocSite) {
  // TODO: Map Linkage Name
  IO.mapRequired("Segment", HeapAllocSite.Header.Segment);
  IO.mapRequired("CallInstructionSize",
                 HeapAllocSite.Header.CallInstructionSize);
  IO.mapRequired("Type", HeapAllocSite.Header.Type);
}

void MappingTraits<FrameCookieSym>::mapping(IO &IO,
                                            FrameCookieSym &FrameCookie) {
  // TODO: Map Linkage Name
  IO.mapRequired("Register", FrameCookie.Header.Register);
  IO.mapRequired("CookieKind", FrameCookie.Header.CookieKind);
  IO.mapRequired("Flags", FrameCookie.Header.Flags);
}

void MappingTraits<CallerSym>::mapping(IO &IO, CallerSym &Caller) {
  // TODO: Correctly handle the ArrayRef in here.
  std::vector<TypeIndex> Indices(Caller.Indices);
  IO.mapRequired("FuncID", Indices);
}

void MappingTraits<UDTSym>::mapping(IO &IO, UDTSym &UDT) {
  IO.mapRequired("Type", UDT.Header.Type);
  IO.mapRequired("UDTName", UDT.Name);
}

void MappingTraits<BuildInfoSym>::mapping(IO &IO, BuildInfoSym &BuildInfo) {
  IO.mapRequired("BuildId", BuildInfo.Header.BuildId);
}

void MappingTraits<BPRelativeSym>::mapping(IO &IO, BPRelativeSym &BPRel) {
  IO.mapRequired("Offset", BPRel.Header.Offset);
  IO.mapRequired("Type", BPRel.Header.Type);
  IO.mapRequired("VarName", BPRel.Name);
}

void MappingTraits<RegRelativeSym>::mapping(IO &IO, RegRelativeSym &RegRel) {
  IO.mapRequired("Offset", RegRel.Header.Offset);
  IO.mapRequired("Type", RegRel.Header.Type);
  IO.mapRequired("Register", RegRel.Header.Register);
  IO.mapRequired("VarName", RegRel.Name);
}

void MappingTraits<ConstantSym>::mapping(IO &IO, ConstantSym &Constant) {
  IO.mapRequired("Type", Constant.Header.Type);
  IO.mapRequired("Value", Constant.Value);
  IO.mapRequired("Name", Constant.Name);
}

void MappingTraits<DataSym>::mapping(IO &IO, DataSym &Data) {
  // TODO: Map linkage name
  IO.mapRequired("Type", Data.Header.Type);
  IO.mapRequired("DisplayName", Data.Name);
}

void MappingTraits<ThreadLocalDataSym>::mapping(IO &IO,
                                                ThreadLocalDataSym &Data) {
  // TODO: Map linkage name
  IO.mapRequired("Type", Data.Header.Type);
  IO.mapRequired("DisplayName", Data.Name);
}
}
}

Error llvm::codeview::yaml::YamlSymbolDumper::visitSymbolBegin(CVSymbol &CVR) {
  YamlIO.mapRequired("Kind", CVR.Type);
  return Error::success();
}
