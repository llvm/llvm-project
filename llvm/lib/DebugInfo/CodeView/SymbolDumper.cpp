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
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolDumpDelegate.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/DebugInfo/CodeView/TypeDumper.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ScopedPrinter.h"

#include <system_error>

using namespace llvm;
using namespace llvm::codeview;

namespace {
/// Use this private dumper implementation to keep implementation details about
/// the visitor out of SymbolDumper.h.
class CVSymbolDumperImpl : public SymbolVisitorCallbacks {
public:
  CVSymbolDumperImpl(CVTypeDumper &CVTD, SymbolDumpDelegate *ObjDelegate,
                     ScopedPrinter &W, bool PrintRecordBytes)
      : CVTD(CVTD), ObjDelegate(ObjDelegate), W(W),
        PrintRecordBytes(PrintRecordBytes), InFunctionScope(false) {}

/// CVSymbolVisitor overrides.
#define SYMBOL_RECORD(EnumName, EnumVal, Name)                                 \
  Error visitKnownRecord(CVSymbol &CVR, Name &Record) override;
#define SYMBOL_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/CVSymbolTypes.def"

  Error visitSymbolBegin(CVSymbol &Record) override;
  Error visitSymbolEnd(CVSymbol &Record) override;
  Error visitUnknownSymbol(CVSymbol &Record) override;

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

Error CVSymbolDumperImpl::visitSymbolBegin(CVSymbol &CVR) {
  return Error::success();
}

Error CVSymbolDumperImpl::visitSymbolEnd(CVSymbol &CVR) {
  if (PrintRecordBytes && ObjDelegate)
    ObjDelegate->printBinaryBlockWithRelocs("SymData", CVR.content());
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, BlockSym &Block) {
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, Thunk32Sym &Thunk) {
  DictScope S(W, "Thunk32");
  W.printNumber("Parent", Thunk.Header.Parent);
  W.printNumber("End", Thunk.Header.End);
  W.printNumber("Next", Thunk.Header.Next);
  W.printNumber("Off", Thunk.Header.Off);
  W.printNumber("Seg", Thunk.Header.Seg);
  W.printNumber("Len", Thunk.Header.Len);
  W.printEnum("Ordinal", Thunk.Header.Ord, getThunkOrdinalNames());
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           TrampolineSym &Tramp) {
  DictScope S(W, "Trampoline");
  W.printEnum("Type", Tramp.Header.Type, getTrampolineNames());
  W.printNumber("Size", Tramp.Header.Size);
  W.printNumber("ThunkOff", Tramp.Header.ThunkOff);
  W.printNumber("TargetOff", Tramp.Header.TargetOff);
  W.printNumber("ThunkSection", Tramp.Header.ThunkSection);
  W.printNumber("TargetSection", Tramp.Header.TargetSection);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, SectionSym &Section) {
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           CoffGroupSym &CoffGroup) {
  DictScope S(W, "COFF Group");
  W.printNumber("Size", CoffGroup.Header.Size);
  W.printFlags("Characteristics", CoffGroup.Header.Characteristics,
               getImageSectionCharacteristicNames(),
               COFF::SectionCharacteristics(0x00F00000));
  W.printNumber("Offset", CoffGroup.Header.Offset);
  W.printNumber("Segment", CoffGroup.Header.Segment);
  W.printString("Name", CoffGroup.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           BPRelativeSym &BPRel) {
  DictScope S(W, "BPRelativeSym");

  W.printNumber("Offset", BPRel.Header.Offset);
  CVTD.printTypeIndex("Type", BPRel.Header.Type);
  W.printString("VarName", BPRel.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           BuildInfoSym &BuildInfo) {
  DictScope S(W, "BuildInfo");

  W.printNumber("BuildId", BuildInfo.Header.BuildId);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           EnvBlockSym &EnvBlock) {
  DictScope S(W, "EnvBlock");

  W.printNumber("Reserved", EnvBlock.Header.Reserved);
  ListScope L(W, "Entries");
  for (auto Entry : EnvBlock.Fields) {
    W.printString(Entry);
  }
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           FileStaticSym &FileStatic) {
  DictScope S(W, "FileStatic");
  W.printNumber("Index", FileStatic.Header.Index);
  W.printNumber("ModFilenameOffset", FileStatic.Header.ModFilenameOffset);
  W.printFlags("Flags", uint16_t(FileStatic.Header.Flags), getLocalFlagNames());
  W.printString("Name", FileStatic.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, ExportSym &Export) {
  DictScope S(W, "Export");
  W.printNumber("Ordinal", Export.Header.Ordinal);
  W.printFlags("Flags", Export.Header.Flags, getExportSymFlagNames());
  W.printString("Name", Export.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           ConstantSym &Constant) {
  DictScope S(W, "Constant");

  CVTD.printTypeIndex("Type", Constant.Header.Type);
  W.printNumber("Value", Constant.Value);
  W.printString("Name", Constant.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, DataSym &Data) {
  DictScope S(W, "DataSym");

  W.printEnum("Kind", uint16_t(CVR.kind()), getSymbolTypeNames());
  StringRef LinkageName;
  if (ObjDelegate) {
    ObjDelegate->printRelocatedField("DataOffset", Data.getRelocationOffset(),
                                     Data.Header.DataOffset, &LinkageName);
  }
  CVTD.printTypeIndex("Type", Data.Header.Type);
  W.printString("DisplayName", Data.Name);
  if (!LinkageName.empty())
    W.printString("LinkageName", LinkageName);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(
    CVSymbol &CVR,
    DefRangeFramePointerRelFullScopeSym &DefRangeFramePointerRelFullScope) {
  DictScope S(W, "DefRangeFramePointerRelFullScope");
  W.printNumber("Offset", DefRangeFramePointerRelFullScope.Header.Offset);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(
    CVSymbol &CVR, DefRangeFramePointerRelSym &DefRangeFramePointerRel) {
  DictScope S(W, "DefRangeFramePointerRel");

  W.printNumber("Offset", DefRangeFramePointerRel.Header.Offset);
  printLocalVariableAddrRange(DefRangeFramePointerRel.Header.Range,
                              DefRangeFramePointerRel.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeFramePointerRel.Gaps);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(
    CVSymbol &CVR, DefRangeRegisterRelSym &DefRangeRegisterRel) {
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(
    CVSymbol &CVR, DefRangeRegisterSym &DefRangeRegister) {
  DictScope S(W, "DefRangeRegister");

  W.printNumber("Register", DefRangeRegister.Header.Register);
  W.printNumber("MayHaveNoName", DefRangeRegister.Header.MayHaveNoName);
  printLocalVariableAddrRange(DefRangeRegister.Header.Range,
                              DefRangeRegister.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeRegister.Gaps);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(
    CVSymbol &CVR, DefRangeSubfieldRegisterSym &DefRangeSubfieldRegister) {
  DictScope S(W, "DefRangeSubfieldRegister");

  W.printNumber("Register", DefRangeSubfieldRegister.Header.Register);
  W.printNumber("MayHaveNoName", DefRangeSubfieldRegister.Header.MayHaveNoName);
  W.printNumber("OffsetInParent",
                DefRangeSubfieldRegister.Header.OffsetInParent);
  printLocalVariableAddrRange(DefRangeSubfieldRegister.Header.Range,
                              DefRangeSubfieldRegister.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeSubfieldRegister.Gaps);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(
    CVSymbol &CVR, DefRangeSubfieldSym &DefRangeSubfield) {
  DictScope S(W, "DefRangeSubfield");

  if (ObjDelegate) {
    StringRef StringTable = ObjDelegate->getStringTable();
    auto ProgramStringTableOffset = DefRangeSubfield.Header.Program;
    if (ProgramStringTableOffset >= StringTable.size())
      return llvm::make_error<CodeViewError>(
          "String table offset outside of bounds of String Table!");
    StringRef Program =
        StringTable.drop_front(ProgramStringTableOffset).split('\0').first;
    W.printString("Program", Program);
  }
  W.printNumber("OffsetInParent", DefRangeSubfield.Header.OffsetInParent);
  printLocalVariableAddrRange(DefRangeSubfield.Header.Range,
                              DefRangeSubfield.getRelocationOffset());
  printLocalVariableAddrGap(DefRangeSubfield.Gaps);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           DefRangeSym &DefRange) {
  DictScope S(W, "DefRange");

  if (ObjDelegate) {
    StringRef StringTable = ObjDelegate->getStringTable();
    auto ProgramStringTableOffset = DefRange.Header.Program;
    if (ProgramStringTableOffset >= StringTable.size())
      return llvm::make_error<CodeViewError>(
          "String table offset outside of bounds of String Table!");
    StringRef Program =
        StringTable.drop_front(ProgramStringTableOffset).split('\0').first;
    W.printString("Program", Program);
  }
  printLocalVariableAddrRange(DefRange.Header.Range,
                              DefRange.getRelocationOffset());
  printLocalVariableAddrGap(DefRange.Gaps);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(
    CVSymbol &CVR, HeapAllocationSiteSym &HeapAllocSite) {
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           InlineSiteSym &InlineSite) {
  DictScope S(W, "InlineSite");

  W.printHex("PtrParent", InlineSite.Header.PtrParent);
  W.printHex("PtrEnd", InlineSite.Header.PtrEnd);
  CVTD.printTypeIndex("Inlinee", InlineSite.Header.Inlinee);

  ListScope BinaryAnnotations(W, "BinaryAnnotations");
  for (auto &Annotation : InlineSite.annotations()) {
    switch (Annotation.OpCode) {
    case BinaryAnnotationsOpCode::Invalid:
      return llvm::make_error<CodeViewError>(
          "Invalid binary annotation opcode!");
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           RegisterSym &Register) {
  DictScope S(W, "RegisterSym");
  W.printNumber("Type", Register.Header.Index);
  W.printEnum("Seg", uint16_t(Register.Header.Register), getRegisterNames());
  W.printString("Name", Register.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, PublicSym32 &Public) {
  DictScope S(W, "PublicSym");
  W.printNumber("Type", Public.Header.Index);
  W.printNumber("Seg", Public.Header.Seg);
  W.printNumber("Off", Public.Header.Off);
  W.printString("Name", Public.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, ProcRefSym &ProcRef) {
  DictScope S(W, "ProcRef");
  W.printNumber("SumName", ProcRef.Header.SumName);
  W.printNumber("SymOffset", ProcRef.Header.SymOffset);
  W.printNumber("Mod", ProcRef.Header.Mod);
  W.printString("Name", ProcRef.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, LabelSym &Label) {
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, LocalSym &Local) {
  DictScope S(W, "Local");

  CVTD.printTypeIndex("Type", Local.Header.Type);
  W.printFlags("Flags", uint16_t(Local.Header.Flags), getLocalFlagNames());
  W.printString("VarName", Local.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, ObjNameSym &ObjName) {
  DictScope S(W, "ObjectName");

  W.printHex("Signature", ObjName.Header.Signature);
  W.printString("ObjectName", ObjName.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, ProcSym &Proc) {
  DictScope S(W, "ProcStart");

  if (InFunctionScope)
    return llvm::make_error<CodeViewError>(
        "Visiting a ProcSym while inside function scope!");

  InFunctionScope = true;

  StringRef LinkageName;
  W.printEnum("Kind", uint16_t(CVR.kind()), getSymbolTypeNames());
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           ScopeEndSym &ScopeEnd) {
  if (CVR.kind() == SymbolKind::S_END)
    DictScope S(W, "BlockEnd");
  else if (CVR.kind() == SymbolKind::S_PROC_ID_END)
    DictScope S(W, "ProcEnd");
  else if (CVR.kind() == SymbolKind::S_INLINESITE_END)
    DictScope S(W, "InlineSiteEnd");

  InFunctionScope = false;
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, CallerSym &Caller) {
  ListScope S(W, CVR.kind() == S_CALLEES ? "Callees" : "Callers");
  for (auto FuncID : Caller.Indices)
    CVTD.printTypeIndex("FuncID", FuncID);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
                                           RegRelativeSym &RegRel) {
  DictScope S(W, "RegRelativeSym");

  W.printHex("Offset", RegRel.Header.Offset);
  CVTD.printTypeIndex("Type", RegRel.Header.Type);
  W.printHex("Register", RegRel.Header.Register);
  W.printString("VarName", RegRel.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR,
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
  return Error::success();
}

Error CVSymbolDumperImpl::visitKnownRecord(CVSymbol &CVR, UDTSym &UDT) {
  DictScope S(W, "UDT");
  CVTD.printTypeIndex("Type", UDT.Header.Type);
  W.printString("UDTName", UDT.Name);
  return Error::success();
}

Error CVSymbolDumperImpl::visitUnknownSymbol(CVSymbol &CVR) {
  DictScope S(W, "UnknownSym");
  W.printEnum("Kind", uint16_t(CVR.kind()), getSymbolTypeNames());
  W.printNumber("Length", CVR.length());
  return Error::success();
}

Error CVSymbolDumper::dump(CVRecord<SymbolKind> &Record) {
  SymbolVisitorCallbackPipeline Pipeline;
  SymbolDeserializer Deserializer(ObjDelegate.get());
  CVSymbolDumperImpl Dumper(CVTD, ObjDelegate.get(), W, PrintRecordBytes);

  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(Dumper);
  CVSymbolVisitor Visitor(Pipeline);
  return Visitor.visitSymbolRecord(Record);
}

Error CVSymbolDumper::dump(const CVSymbolArray &Symbols) {
  SymbolVisitorCallbackPipeline Pipeline;
  SymbolDeserializer Deserializer(ObjDelegate.get());
  CVSymbolDumperImpl Dumper(CVTD, ObjDelegate.get(), W, PrintRecordBytes);

  Pipeline.addCallbackToPipeline(Deserializer);
  Pipeline.addCallbackToPipeline(Dumper);
  CVSymbolVisitor Visitor(Pipeline);
  return Visitor.visitSymbolStream(Symbols);
}
