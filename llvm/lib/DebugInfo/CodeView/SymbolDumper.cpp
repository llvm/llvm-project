//===-- SymbolDumper.cpp - CodeView symbol info dumper ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"
#include "llvm/DebugInfo/CodeView/SymbolDumpDelegate.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeDumper.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/ScopedPrinter.h"

#include <system_error>

using namespace llvm;
using namespace llvm::codeview;

namespace {
/// Use this private dumper implementation to keep implementation details about
/// the visitor out of SymbolDumper.h.
class CVSymbolDumperImpl : public CVSymbolVisitor<CVSymbolDumperImpl> {
public:
  CVSymbolDumperImpl(CVTypeDumper &CVTD, SymbolDumpDelegate *ObjDelegate,
                     ScopedPrinter &W, bool PrintRecordBytes)
      : CVSymbolVisitor(ObjDelegate), CVTD(CVTD), ObjDelegate(ObjDelegate),
        W(W), PrintRecordBytes(PrintRecordBytes), InFunctionScope(false) {}

/// CVSymbolVisitor overrides.
#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  void visit##Name(SymbolKind Kind, Name &Record);
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CVSymbolTypes.def"

  void visitSymbolBegin(SymbolKind Kind, ArrayRef<uint8_t> Data);
  void visitSymbolEnd(SymbolKind Kind, ArrayRef<uint8_t> OriginalSymData);
  void visitUnknownSymbol(SymbolKind Kind, ArrayRef<uint8_t> Data);

private:
  void printLocalVariableAddrRange(const LocalVariableAddrRange &Range,
                                   uint32_t RelocationOffset);
  void printLocalVariableAddrGap(ArrayRef<LocalVariableAddrGap> Gaps);

  CVTypeDumper &CVTD;
  SymbolDumpDelegate *ObjDelegate;
  ScopedPrinter &W;

  bool PrintRecordBytes;
  bool InFunctionScope;
};
}

void CVSymbolDumperImpl::printLocalVariableAddrRange(
    const LocalVariableAddrRange &Range, uint32_t RelocationOffset) {
  DictScope S(W, "LocalVariableAddrRange");
  if (ObjDelegate)
    ObjDelegate->printRelocatedField("OffsetStart", RelocationOffset,
                                     Range.OffsetStart);
  W.printHex("ISectStart", Range.ISectStart);
  W.printHex("Range", Range.Range);
}

void CVSymbolDumperImpl::printLocalVariableAddrGap(
    ArrayRef<LocalVariableAddrGap> Gaps) {
  for (auto &Gap : Gaps) {
    ListScope S(W, "LocalVariableAddrGap");
    W.printHex("GapStartOffset", Gap.GapStartOffset);
    W.printHex("Range", Gap.Range);
  }
}

void CVSymbolDumperImpl::visitSymbolBegin(SymbolKind Kind,
                                          ArrayRef<uint8_t> Data) {}

void CVSymbolDumperImpl::visitSymbolEnd(SymbolKind Kind,
                                        ArrayRef<uint8_t> OriginalSymData) {
  if (PrintRecordBytes && ObjDelegate)
    ObjDelegate->printBinaryBlockWithRelocs("SymData", OriginalSymData);
}

void CVSymbolDumperImpl::visitBlockSym(SymbolKind Kind, BlockSym &Block) {
  DictScope S(W, "BlockStart");

  StringRef LinkageName;
  W.printHex("PtrParent", Block.Header.PtrParent);
  W.printHex("PtrEnd", Block.Header.PtrEnd);
  W.printHex("CodeSize", Block.Header.CodeSize);
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField("CodeOffset", Block.getRelocationOffset(),
                                     Block.Header.CodeOffset, &LinkageName);
  }
  W.printHex("Segment", Block.Header.Segment);
  W.printString("BlockName", Block.Name);
  W.printString("LinkageName", LinkageName);
}

void CVSymbolDumperImpl::visitThunk32Sym(SymbolKind Kind, Thunk32Sym &Thunk) {
  DictScope S(W, "Thunk32");
  W.printNumber("Parent", Thunk.Header.Parent);
  W.printNumber("End", Thunk.Header.End);
  W.printNumber("Next", Thunk.Header.Next);
  W.printNumber("Off", Thunk.Header.Off);
  W.printNumber("Seg", Thunk.Header.Seg);
  W.printNumber("Len", Thunk.Header.Len);
  W.printEnum("Ordinal", Thunk.Header.Ord, getThunkOrdinalNames());
}

void CVSymbolDumperImpl::visitTrampolineSym(SymbolKind Kind,
                                            TrampolineSym &Tramp) {
  DictScope S(W, "Trampoline");
  W.printEnum("Type", Tramp.Header.Type, getTrampolineNames());
  W.printNumber("Size", Tramp.Header.Size);
  W.printNumber("ThunkOff", Tramp.Header.ThunkOff);
  W.printNumber("TargetOff", Tramp.Header.TargetOff);
  W.printNumber("ThunkSection", Tramp.Header.ThunkSection);
  W.printNumber("TargetSection", Tramp.Header.TargetSection);
}

void CVSymbolDumperImpl::visitSectionSym(SymbolKind Kind, SectionSym &Section) {
  DictScope S(W, "Section");
  W.printNumber("SectionNumber", Section.Header.SectionNumber);
  W.printNumber("Alignment", Section.Header.Alignment);
  W.printNumber("Reserved", Section.Header.Reserved);
  W.printNumber("Rva", Section.Header.Rva);
  W.printNumber("Length", Section.Header.Length);
  W.printFlags("Characteristics", Section.Header.Characteristics,
               getImageSectionCharacteristicNames(),
               COFF::SectionCharacteristics(0x00F00000));

  W.printString("Name", Section.Name);
}

void CVSymbolDumperImpl::visitCoffGroupSym(SymbolKind Kind,
                                           CoffGroupSym &CoffGroup) {
  DictScope S(W, "COFF Group");
  W.printNumber("Size", CoffGroup.Header.Size);
  W.printFlags("Characteristics", CoffGroup.Header.Characteristics,
               getImageSectionCharacteristicNames(),
               COFF::SectionCharacteristics(0x00F00000));
  W.printNumber("Offset", CoffGroup.Header.Offset);
  W.printNumber("Segment", CoffGroup.Header.Segment);
  W.printString("Name", CoffGroup.Name);
}

void CVSymbolDumperImpl::visitBPRelativeSym(SymbolKind Kind,
                                            BPRelativeSym &BPRel) {
  DictScope S(W, "BPRelativeSym");

  W.printNumber("Offset", BPRel.Header.Offset);
  CVTD.printTypeIndex("Type", BPRel.Header.Type);
  W.printString("VarName", BPRel.Name);
}

void CVSymbolDumperImpl::visitBuildInfoSym(SymbolKind Kind,
                                           BuildInfoSym &BuildInfo) {
  DictScope S(W, "BuildInfo");

  W.printNumber("BuildId", BuildInfo.Header.BuildId);
}

void CVSymbolDumperImpl::visitCallSiteInfoSym(SymbolKind Kind,
                                              CallSiteInfoSym &CallSiteInfo) {
  DictScope S(W, "CallSiteInfo");

  StringRef LinkageName;
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField(
        "CodeOffset", CallSiteInfo.getRelocationOffset(),
        CallSiteInfo.Header.CodeOffset, &LinkageName);
  }
  W.printHex("Segment", CallSiteInfo.Header.Segment);
  W.printHex("Reserved", CallSiteInfo.Header.Reserved);
  CVTD.printTypeIndex("Type", CallSiteInfo.Header.Type);
  if (!LinkageName.empty())
    W.printString("LinkageName", LinkageName);
}

void CVSymbolDumperImpl::visitEnvBlockSym(SymbolKind Kind,
                                          EnvBlockSym &EnvBlock) {
  DictScope S(W, "EnvBlock");

  W.printNumber("Reserved", EnvBlock.Header.Reserved);
  ListScope L(W, "Entries");
  for (auto Entry : EnvBlock.Fields) {
    W.printString(Entry);
  }
}

void CVSymbolDumperImpl::visitFileStaticSym(SymbolKind Kind,
                                            FileStaticSym &FileStatic) {
  DictScope S(W, "FileStatic");
  W.printNumber("Index", FileStatic.Header.Index);
  W.printNumber("ModFilenameOffset", FileStatic.Header.ModFilenameOffset);
  W.printFlags("Flags", uint16_t(FileStatic.Header.Flags), getLocalFlagNames());
  W.printString("Name", FileStatic.Name);
}

void CVSymbolDumperImpl::visitExportSym(SymbolKind Kind, ExportSym &Export) {
  DictScope S(W, "Export");
  W.printNumber("Ordinal", Export.Header.Ordinal);
  W.printFlags("Flags", Export.Header.Flags, getExportSymFlagNames());
  W.printString("Name", Export.Name);
}

void CVSymbolDumperImpl::visitCompile2Sym(SymbolKind Kind,
                                          Compile2Sym &Compile2) {
  DictScope S(W, "CompilerFlags2");

  W.printEnum("Language", Compile2.Header.getLanguage(),
              getSourceLanguageNames());
  W.printFlags("Flags", Compile2.Header.flags & ~0xff,
               getCompileSym2FlagNames());
  W.printEnum("Machine", unsigned(Compile2.Header.Machine), getCPUTypeNames());
  std::string FrontendVersion;
  {
    raw_string_ostream Out(FrontendVersion);
    Out << Compile2.Header.VersionFrontendMajor << '.'
        << Compile2.Header.VersionFrontendMinor << '.'
        << Compile2.Header.VersionFrontendBuild;
  }
  std::string BackendVersion;
  {
    raw_string_ostream Out(BackendVersion);
    Out << Compile2.Header.VersionBackendMajor << '.'
        << Compile2.Header.VersionBackendMinor << '.'
        << Compile2.Header.VersionBackendBuild;
  }
  W.printString("FrontendVersion", FrontendVersion);
  W.printString("BackendVersion", BackendVersion);
  W.printString("VersionName", Compile2.Version);
}

void CVSymbolDumperImpl::visitCompile3Sym(SymbolKind Kind,
                                          Compile3Sym &Compile3) {
  DictScope S(W, "CompilerFlags3");

  W.printEnum("Language", Compile3.Header.getLanguage(),
              getSourceLanguageNames());
  W.printFlags("Flags", Compile3.Header.flags & ~0xff,
               getCompileSym3FlagNames());
  W.printEnum("Machine", unsigned(Compile3.Header.Machine), getCPUTypeNames());
  std::string FrontendVersion;
  {
    raw_string_ostream Out(FrontendVersion);
    Out << Compile3.Header.VersionFrontendMajor << '.'
        << Compile3.Header.VersionFrontendMinor << '.'
        << Compile3.Header.VersionFrontendBuild << '.'
        << Compile3.Header.VersionFrontendQFE;
  }
  std::string BackendVersion;
  {
    raw_string_ostream Out(BackendVersion);
    Out << Compile3.Header.VersionBackendMajor << '.'
        << Compile3.Header.VersionBackendMinor << '.'
        << Compile3.Header.VersionBackendBuild << '.'
        << Compile3.Header.VersionBackendQFE;
  }
  W.printString("FrontendVersion", FrontendVersion);
  W.printString("BackendVersion", BackendVersion);
  W.printString("VersionName", Compile3.Version);
}

void CVSymbolDumperImpl::visitConstantSym(SymbolKind Kind,
                                          ConstantSym &Constant) {
  DictScope S(W, "Constant");

  CVTD.printTypeIndex("Type", Constant.Header.Type);
  W.printNumber("Value", Constant.Value);
  W.printString("Name", Constant.Name);
}

void CVSymbolDumperImpl::visitDataSym(SymbolKind Kind, DataSym &Data) {
  DictScope S(W, "DataSym");

  StringRef LinkageName;
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField("DataOffset", Data.getRelocationOffset(),
                                     Data.Header.DataOffset, &LinkageName);
  }
  CVTD.printTypeIndex("Type", Data.Header.Type);
  W.printString("DisplayName", Data.Name);
  if (!LinkageName.empty())
    W.printString("LinkageName", LinkageName);
}

void CVSymbolDumperImpl::visitDefRangeFramePointerRelFullScopeSym(
    SymbolKind Kind,
    DefRangeFramePointerRelFullScopeSym &DefRangeFramePointerRelFullScope) {
  DictScope S(W, "DefRangeFramePointerRelFullScope");
  W.printNumber("Offset", DefRangeFramePointerRelFullScope.Header.Offset);
}

void CVSymbolDumperImpl::visitDefRangeFramePointerRelSym(
    SymbolKind Kind, DefRangeFramePointerRelSym &DefRangeFramePointerRel) {
  DictScope S(W, "DefRangeFramePointerRel");

  W.printNumber("Offset", DefRangeFramePointerRel.Header.Offset);
  printLocalVariableAddrRange(DefRangeFramePointerRel.Header.Range,
                              DefRangeFramePointerRel.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeFramePointerRel.Gaps);
}

void CVSymbolDumperImpl::visitDefRangeRegisterRelSym(
    SymbolKind Kind, DefRangeRegisterRelSym &DefRangeRegisterRel) {
  DictScope S(W, "DefRangeRegisterRel");

  W.printNumber("BaseRegister", DefRangeRegisterRel.Header.BaseRegister);
  W.printBoolean("HasSpilledUDTMember",
                 DefRangeRegisterRel.hasSpilledUDTMember());
  W.printNumber("OffsetInParent", DefRangeRegisterRel.offsetInParent());
  W.printNumber("BasePointerOffset",
                DefRangeRegisterRel.Header.BasePointerOffset);
  printLocalVariableAddrRange(DefRangeRegisterRel.Header.Range,
                              DefRangeRegisterRel.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeRegisterRel.Gaps);
}

void CVSymbolDumperImpl::visitDefRangeRegisterSym(
    SymbolKind Kind, DefRangeRegisterSym &DefRangeRegister) {
  DictScope S(W, "DefRangeRegister");

  W.printNumber("Register", DefRangeRegister.Header.Register);
  W.printNumber("MayHaveNoName", DefRangeRegister.Header.MayHaveNoName);
  printLocalVariableAddrRange(DefRangeRegister.Header.Range,
                              DefRangeRegister.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeRegister.Gaps);
}

void CVSymbolDumperImpl::visitDefRangeSubfieldRegisterSym(
    SymbolKind Kind, DefRangeSubfieldRegisterSym &DefRangeSubfieldRegister) {
  DictScope S(W, "DefRangeSubfieldRegister");

  W.printNumber("Register", DefRangeSubfieldRegister.Header.Register);
  W.printNumber("MayHaveNoName", DefRangeSubfieldRegister.Header.MayHaveNoName);
  W.printNumber("OffsetInParent",
                DefRangeSubfieldRegister.Header.OffsetInParent);
  printLocalVariableAddrRange(DefRangeSubfieldRegister.Header.Range,
                              DefRangeSubfieldRegister.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeSubfieldRegister.Gaps);
}

void CVSymbolDumperImpl::visitDefRangeSubfieldSym(
    SymbolKind Kind, DefRangeSubfieldSym &DefRangeSubfield) {
  DictScope S(W, "DefRangeSubfield");

  if (ObjDelegate) {
    StringRef StringTable = ObjDelegate->getStringTable();
    auto ProgramStringTableOffset = DefRangeSubfield.Header.Program;
    if (ProgramStringTableOffset >= StringTable.size())
      return parseError();
    StringRef Program =
        StringTable.drop_front(ProgramStringTableOffset).split('\0').first;
    W.printString("Program", Program);
  }
  W.printNumber("OffsetInParent", DefRangeSubfield.Header.OffsetInParent);
  printLocalVariableAddrRange(DefRangeSubfield.Header.Range,
                              DefRangeSubfield.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeSubfield.Gaps);
}

void CVSymbolDumperImpl::visitDefRangeSym(SymbolKind Kind,
                                          DefRangeSym &DefRange) {
  DictScope S(W, "DefRange");

  if (ObjDelegate) {
    StringRef StringTable = ObjDelegate->getStringTable();
    auto ProgramStringTableOffset = DefRange.Header.Program;
    if (ProgramStringTableOffset >= StringTable.size())
      return parseError();
    StringRef Program =
        StringTable.drop_front(ProgramStringTableOffset).split('\0').first;
    W.printString("Program", Program);
  }
  printLocalVariableAddrRange(DefRange.Header.Range,
                              DefRange.getRelocationOffset());
  printLocalVariableAddrGap(DefRange.Gaps);
}

void CVSymbolDumperImpl::visitFrameCookieSym(SymbolKind Kind,
                                             FrameCookieSym &FrameCookie) {
  DictScope S(W, "FrameCookie");

  StringRef LinkageName;
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField(
        "CodeOffset", FrameCookie.getRelocationOffset(),
        FrameCookie.Header.CodeOffset, &LinkageName);
  }
  W.printHex("Register", FrameCookie.Header.Register);
  W.printEnum("CookieKind", uint16_t(FrameCookie.Header.CookieKind),
              getFrameCookieKindNames());
  W.printHex("Flags", FrameCookie.Header.Flags);
}

void CVSymbolDumperImpl::visitFrameProcSym(SymbolKind Kind,
                                           FrameProcSym &FrameProc) {
  DictScope S(W, "FrameProc");

  W.printHex("TotalFrameBytes", FrameProc.Header.TotalFrameBytes);
  W.printHex("PaddingFrameBytes", FrameProc.Header.PaddingFrameBytes);
  W.printHex("OffsetToPadding", FrameProc.Header.OffsetToPadding);
  W.printHex("BytesOfCalleeSavedRegisters",
             FrameProc.Header.BytesOfCalleeSavedRegisters);
  W.printHex("OffsetOfExceptionHandler",
             FrameProc.Header.OffsetOfExceptionHandler);
  W.printHex("SectionIdOfExceptionHandler",
             FrameProc.Header.SectionIdOfExceptionHandler);
  W.printFlags("Flags", FrameProc.Header.Flags, getFrameProcSymFlagNames());
}

void CVSymbolDumperImpl::visitHeapAllocationSiteSym(
    SymbolKind Kind, HeapAllocationSiteSym &HeapAllocSite) {
  DictScope S(W, "HeapAllocationSite");

  StringRef LinkageName;
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField(
        "CodeOffset", HeapAllocSite.getRelocationOffset(),
        HeapAllocSite.Header.CodeOffset, &LinkageName);
  }
  W.printHex("Segment", HeapAllocSite.Header.Segment);
  W.printHex("CallInstructionSize", HeapAllocSite.Header.CallInstructionSize);
  CVTD.printTypeIndex("Type", HeapAllocSite.Header.Type);
  if (!LinkageName.empty())
    W.printString("LinkageName", LinkageName);
}

void CVSymbolDumperImpl::visitInlineSiteSym(SymbolKind Kind,
                                            InlineSiteSym &InlineSite) {
  DictScope S(W, "InlineSite");

  W.printHex("PtrParent", InlineSite.Header.PtrParent);
  W.printHex("PtrEnd", InlineSite.Header.PtrEnd);
  CVTD.printTypeIndex("Inlinee", InlineSite.Header.Inlinee);

  ListScope BinaryAnnotations(W, "BinaryAnnotations");
  for (auto &Annotation : InlineSite.annotations()) {
    switch (Annotation.OpCode) {
    case BinaryAnnotationsOpCode::Invalid:
      return parseError();
    case BinaryAnnotationsOpCode::CodeOffset:
    case BinaryAnnotationsOpCode::ChangeCodeOffset:
    case BinaryAnnotationsOpCode::ChangeCodeLength:
      W.printHex(Annotation.Name, Annotation.U1);
      break;
    case BinaryAnnotationsOpCode::ChangeCodeOffsetBase:
    case BinaryAnnotationsOpCode::ChangeLineEndDelta:
    case BinaryAnnotationsOpCode::ChangeRangeKind:
    case BinaryAnnotationsOpCode::ChangeColumnStart:
    case BinaryAnnotationsOpCode::ChangeColumnEnd:
      W.printNumber(Annotation.Name, Annotation.U1);
      break;
    case BinaryAnnotationsOpCode::ChangeLineOffset:
    case BinaryAnnotationsOpCode::ChangeColumnEndDelta:
      W.printNumber(Annotation.Name, Annotation.S1);
      break;
    case BinaryAnnotationsOpCode::ChangeFile:
      if (ObjDelegate) {
        W.printHex("ChangeFile",
                   ObjDelegate->getFileNameForFileOffset(Annotation.U1),
                   Annotation.U1);
      } else {
        W.printHex("ChangeFile", Annotation.U1);
      }

      break;
    case BinaryAnnotationsOpCode::ChangeCodeOffsetAndLineOffset: {
      W.startLine() << "ChangeCodeOffsetAndLineOffset: {CodeOffset: "
                    << W.hex(Annotation.U1) << ", LineOffset: " << Annotation.S1
                    << "}\n";
      break;
    }
    case BinaryAnnotationsOpCode::ChangeCodeLengthAndCodeOffset: {
      W.startLine() << "ChangeCodeLengthAndCodeOffset: {CodeOffset: "
                    << W.hex(Annotation.U2)
                    << ", Length: " << W.hex(Annotation.U1) << "}\n";
      break;
    }
    }
  }
}

void CVSymbolDumperImpl::visitRegisterSym(SymbolKind Kind,
                                          RegisterSym &Register) {
  DictScope S(W, "RegisterSym");
  W.printNumber("Type", Register.Header.Index);
  W.printEnum("Seg", uint16_t(Register.Header.Register), getRegisterNames());
  W.printString("Name", Register.Name);
}

void CVSymbolDumperImpl::visitPublicSym32(SymbolKind Kind,
                                          PublicSym32 &Public) {
  DictScope S(W, "PublicSym");
  W.printNumber("Type", Public.Header.Index);
  W.printNumber("Seg", Public.Header.Seg);
  W.printNumber("Off", Public.Header.Off);
  W.printString("Name", Public.Name);
}

void CVSymbolDumperImpl::visitProcRefSym(SymbolKind Kind, ProcRefSym &ProcRef) {
  DictScope S(W, "ProcRef");
  W.printNumber("SumName", ProcRef.Header.SumName);
  W.printNumber("SymOffset", ProcRef.Header.SymOffset);
  W.printNumber("Mod", ProcRef.Header.Mod);
  W.printString("Name", ProcRef.Name);
}

void CVSymbolDumperImpl::visitLabelSym(SymbolKind Kind, LabelSym &Label) {
  DictScope S(W, "Label");

  StringRef LinkageName;
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField("CodeOffset", Label.getRelocationOffset(),
                                     Label.Header.CodeOffset, &LinkageName);
  }
  W.printHex("Segment", Label.Header.Segment);
  W.printHex("Flags", Label.Header.Flags);
  W.printFlags("Flags", Label.Header.Flags, getProcSymFlagNames());
  W.printString("DisplayName", Label.Name);
  if (!LinkageName.empty())
    W.printString("LinkageName", LinkageName);
}

void CVSymbolDumperImpl::visitLocalSym(SymbolKind Kind, LocalSym &Local) {
  DictScope S(W, "Local");

  CVTD.printTypeIndex("Type", Local.Header.Type);
  W.printFlags("Flags", uint16_t(Local.Header.Flags), getLocalFlagNames());
  W.printString("VarName", Local.Name);
}

void CVSymbolDumperImpl::visitObjNameSym(SymbolKind Kind, ObjNameSym &ObjName) {
  DictScope S(W, "ObjectName");

  W.printHex("Signature", ObjName.Header.Signature);
  W.printString("ObjectName", ObjName.Name);
}

void CVSymbolDumperImpl::visitProcSym(SymbolKind Kind, ProcSym &Proc) {
  DictScope S(W, "ProcStart");

  if (InFunctionScope)
    return parseError();

  InFunctionScope = true;

  StringRef LinkageName;
  W.printHex("PtrParent", Proc.Header.PtrParent);
  W.printHex("PtrEnd", Proc.Header.PtrEnd);
  W.printHex("PtrNext", Proc.Header.PtrNext);
  W.printHex("CodeSize", Proc.Header.CodeSize);
  W.printHex("DbgStart", Proc.Header.DbgStart);
  W.printHex("DbgEnd", Proc.Header.DbgEnd);
  CVTD.printTypeIndex("FunctionType", Proc.Header.FunctionType);
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField("CodeOffset", Proc.getRelocationOffset(),
                                     Proc.Header.CodeOffset, &LinkageName);
  }
  W.printHex("Segment", Proc.Header.Segment);
  W.printFlags("Flags", static_cast<uint8_t>(Proc.Header.Flags),
               getProcSymFlagNames());
  W.printString("DisplayName", Proc.Name);
  if (!LinkageName.empty())
    W.printString("LinkageName", LinkageName);
}

void CVSymbolDumperImpl::visitScopeEndSym(SymbolKind Kind,
                                          ScopeEndSym &ScopeEnd) {
  if (Kind == SymbolKind::S_END)
    DictScope S(W, "BlockEnd");
  else if (Kind == SymbolKind::S_PROC_ID_END)
    DictScope S(W, "ProcEnd");
  else if (Kind == SymbolKind::S_INLINESITE_END)
    DictScope S(W, "InlineSiteEnd");

  InFunctionScope = false;
}

void CVSymbolDumperImpl::visitCallerSym(SymbolKind Kind, CallerSym &Caller) {
  ListScope S(W, Kind == S_CALLEES ? "Callees" : "Callers");
  for (auto FuncID : Caller.Indices)
    CVTD.printTypeIndex("FuncID", FuncID);
}

void CVSymbolDumperImpl::visitRegRelativeSym(SymbolKind Kind,
                                             RegRelativeSym &RegRel) {
  DictScope S(W, "RegRelativeSym");

  W.printHex("Offset", RegRel.Header.Offset);
  CVTD.printTypeIndex("Type", RegRel.Header.Type);
  W.printHex("Register", RegRel.Header.Register);
  W.printString("VarName", RegRel.Name);
}

void CVSymbolDumperImpl::visitThreadLocalDataSym(SymbolKind Kind,
                                                 ThreadLocalDataSym &Data) {
  DictScope S(W, "ThreadLocalDataSym");

  StringRef LinkageName;
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField("DataOffset", Data.getRelocationOffset(),
                                     Data.Header.DataOffset, &LinkageName);
  }
  CVTD.printTypeIndex("Type", Data.Header.Type);
  W.printString("DisplayName", Data.Name);
  if (!LinkageName.empty())
    W.printString("LinkageName", LinkageName);
}

void CVSymbolDumperImpl::visitUDTSym(SymbolKind Kind, UDTSym &UDT) {
  DictScope S(W, "UDT");
  CVTD.printTypeIndex("Type", UDT.Header.Type);
  W.printString("UDTName", UDT.Name);
}

void CVSymbolDumperImpl::visitUnknownSymbol(SymbolKind Kind,
                                            ArrayRef<uint8_t> Data) {
  DictScope S(W, "UnknownSym");
  W.printEnum("Kind", uint16_t(Kind), getSymbolTypeNames());
  W.printNumber("Length", uint32_t(Data.size()));
}

bool CVSymbolDumper::dump(const CVRecord<SymbolKind> &Record) {
  CVSymbolDumperImpl Dumper(CVTD, ObjDelegate.get(), W, PrintRecordBytes);
  Dumper.visitSymbolRecord(Record);
  return !Dumper.hadError();
}

bool CVSymbolDumper::dump(const CVSymbolArray &Symbols) {
  CVSymbolDumperImpl Dumper(CVTD, ObjDelegate.get(), W, PrintRecordBytes);
  Dumper.visitSymbolStream(Symbols);
  return !Dumper.hadError();
}
