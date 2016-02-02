//===-- llvm/lib/CodeGen/AsmPrinter/CodeViewDebug.cpp --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Microsoft CodeView debug info.
//
//===----------------------------------------------------------------------===//

#include "CodeViewDebug.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/Line.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/COFF.h"

using namespace llvm::codeview;

namespace llvm {

StringRef CodeViewDebug::getFullFilepath(const DIFile *File) {
  std::string &Filepath = FileToFilepathMap[File];
  if (!Filepath.empty())
    return Filepath;

  StringRef Dir = File->getDirectory(), Filename = File->getFilename();

  // Clang emits directory and relative filename info into the IR, but CodeView
  // operates on full paths.  We could change Clang to emit full paths too, but
  // that would increase the IR size and probably not needed for other users.
  // For now, just concatenate and canonicalize the path here.
  if (Filename.find(':') == 1)
    Filepath = Filename;
  else
    Filepath = (Dir + "\\" + Filename).str();

  // Canonicalize the path.  We have to do it textually because we may no longer
  // have access the file in the filesystem.
  // First, replace all slashes with backslashes.
  std::replace(Filepath.begin(), Filepath.end(), '/', '\\');

  // Remove all "\.\" with "\".
  size_t Cursor = 0;
  while ((Cursor = Filepath.find("\\.\\", Cursor)) != std::string::npos)
    Filepath.erase(Cursor, 2);

  // Replace all "\XXX\..\" with "\".  Don't try too hard though as the original
  // path should be well-formatted, e.g. start with a drive letter, etc.
  Cursor = 0;
  while ((Cursor = Filepath.find("\\..\\", Cursor)) != std::string::npos) {
    // Something's wrong if the path starts with "\..\", abort.
    if (Cursor == 0)
      break;

    size_t PrevSlash = Filepath.rfind('\\', Cursor - 1);
    if (PrevSlash == std::string::npos)
      // Something's wrong, abort.
      break;

    Filepath.erase(PrevSlash, Cursor + 3 - PrevSlash);
    // The next ".." might be following the one we've just erased.
    Cursor = PrevSlash;
  }

  // Remove all duplicate backslashes.
  Cursor = 0;
  while ((Cursor = Filepath.find("\\\\", Cursor)) != std::string::npos)
    Filepath.erase(Cursor, 1);

  return Filepath;
}

unsigned CodeViewDebug::maybeRecordFile(const DIFile *F) {
  unsigned NextId = FileIdMap.size() + 1;
  auto Insertion = FileIdMap.insert(std::make_pair(F, NextId));
  if (Insertion.second) {
    // We have to compute the full filepath and emit a .cv_file directive.
    StringRef FullPath = getFullFilepath(F);
    NextId = Asm->OutStreamer->EmitCVFileDirective(NextId, FullPath);
    assert(NextId == FileIdMap.size() && ".cv_file directive failed");
  }
  return Insertion.first->second;
}

CodeViewDebug::InlineSite &CodeViewDebug::getInlineSite(const DILocation *Loc) {
  const DILocation *InlinedAt = Loc->getInlinedAt();
  auto Insertion = CurFn->InlineSites.insert({InlinedAt, InlineSite()});
  if (Insertion.second) {
    InlineSite &Site = Insertion.first->second;
    Site.SiteFuncId = NextFuncId++;
    Site.Inlinee = Loc->getScope()->getSubprogram();
    InlinedSubprograms.insert(Loc->getScope()->getSubprogram());
  }
  return Insertion.first->second;
}

void CodeViewDebug::maybeRecordLocation(DebugLoc DL,
                                        const MachineFunction *MF) {
  // Skip this instruction if it has the same location as the previous one.
  if (DL == CurFn->LastLoc)
    return;

  const DIScope *Scope = DL.get()->getScope();
  if (!Scope)
    return;

  // Skip this line if it is longer than the maximum we can record.
  LineInfo LI(DL.getLine(), DL.getLine(), /*IsStatement=*/true);
  if (LI.getStartLine() != DL.getLine() || LI.isAlwaysStepInto() ||
      LI.isNeverStepInto())
    return;

  ColumnInfo CI(DL.getCol(), /*EndColumn=*/0);
  if (CI.getStartColumn() != DL.getCol())
    return;

  if (!CurFn->HaveLineInfo)
    CurFn->HaveLineInfo = true;
  unsigned FileId = 0;
  if (CurFn->LastLoc.get() && CurFn->LastLoc->getFile() == DL->getFile())
    FileId = CurFn->LastFileId;
  else
    FileId = CurFn->LastFileId = maybeRecordFile(DL->getFile());
  CurFn->LastLoc = DL;

  unsigned FuncId = CurFn->FuncId;
  if (const DILocation *Loc = DL->getInlinedAt()) {
    // If this location was actually inlined from somewhere else, give it the ID
    // of the inline call site.
    FuncId = getInlineSite(DL.get()).SiteFuncId;
    // Ensure we have links in the tree of inline call sites.
    const DILocation *ChildLoc = nullptr;
    while (Loc->getInlinedAt()) {
      InlineSite &Site = getInlineSite(Loc);
      if (ChildLoc) {
        // Record the child inline site if not already present.
        auto B = Site.ChildSites.begin(), E = Site.ChildSites.end();
        if (std::find(B, E, Loc) != E)
          break;
        Site.ChildSites.push_back(Loc);
      }
      ChildLoc = Loc;
    }
  }

  Asm->OutStreamer->EmitCVLocDirective(FuncId, FileId, DL.getLine(),
                                       DL.getCol(), /*PrologueEnd=*/false,
                                       /*IsStmt=*/false, DL->getFilename());
}

CodeViewDebug::CodeViewDebug(AsmPrinter *AP)
    : Asm(nullptr), CurFn(nullptr) {
  MachineModuleInfo *MMI = AP->MMI;

  // If module doesn't have named metadata anchors or COFF debug section
  // is not available, skip any debug info related stuff.
  if (!MMI->getModule()->getNamedMetadata("llvm.dbg.cu") ||
      !AP->getObjFileLowering().getCOFFDebugSymbolsSection())
    return;

  // Tell MMI that we have debug info.
  MMI->setDebugInfoAvailability(true);
  Asm = AP;
}

void CodeViewDebug::endModule() {
  if (FnDebugInfo.empty())
    return;

  emitTypeInformation();

  // FIXME: For functions that are comdat, we should emit separate .debug$S
  // sections that are comdat associative with the main function instead of
  // having one big .debug$S section.
  assert(Asm != nullptr);
  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getCOFFDebugSymbolsSection());
  Asm->OutStreamer->AddComment("Debug section magic");
  Asm->EmitInt32(COFF::DEBUG_SECTION_MAGIC);

  // The COFF .debug$S section consists of several subsections, each starting
  // with a 4-byte control code (e.g. 0xF1, 0xF2, etc) and then a 4-byte length
  // of the payload followed by the payload itself.  The subsections are 4-byte
  // aligned.

  // Make a subsection for all the inlined subprograms.
  emitInlineeLinesSubsection();

  // Emit per-function debug information.
  for (auto &P : FnDebugInfo)
    emitDebugInfoForFunction(P.first, P.second);

  // This subsection holds a file index to offset in string table table.
  Asm->OutStreamer->AddComment("File index to string table offset subsection");
  Asm->OutStreamer->EmitCVFileChecksumsDirective();

  // This subsection holds the string table.
  Asm->OutStreamer->AddComment("String table");
  Asm->OutStreamer->EmitCVStringTableDirective();

  clear();
}

void CodeViewDebug::emitTypeInformation() {
  // Start the .debug$T section with 0x4.
  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getCOFFDebugTypesSection());
  Asm->OutStreamer->AddComment("Debug section magic");
  Asm->EmitInt32(COFF::DEBUG_SECTION_MAGIC);

  NamedMDNode *CU_Nodes =
      Asm->MMI->getModule()->getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes)
    return;

  // This type info currently only holds function ids for use with inline call
  // frame info. All functions are assigned a simple 'void ()' type. Emit that
  // type here.
  TypeIndex ArgListIdx = getNextTypeIndex();
  Asm->OutStreamer->AddComment("Type record length");
  Asm->EmitInt16(2 + sizeof(ArgList));
  Asm->OutStreamer->AddComment("Leaf type: LF_ARGLIST");
  Asm->EmitInt16(LF_ARGLIST);
  Asm->OutStreamer->AddComment("Number of arguments");
  Asm->EmitInt32(0);

  TypeIndex VoidProcIdx = getNextTypeIndex();
  Asm->OutStreamer->AddComment("Type record length");
  Asm->EmitInt16(2 + sizeof(ProcedureType));
  Asm->OutStreamer->AddComment("Leaf type: LF_PROCEDURE");
  Asm->EmitInt16(LF_PROCEDURE);
  Asm->OutStreamer->AddComment("Return type index");
  Asm->EmitInt32(TypeIndex::Void().getIndex());
  Asm->OutStreamer->AddComment("Calling convention");
  Asm->EmitInt8(char(CallingConvention::NearC));
  Asm->OutStreamer->AddComment("Function options");
  Asm->EmitInt8(char(FunctionOptions::None));
  Asm->OutStreamer->AddComment("# of parameters");
  Asm->EmitInt16(0);
  Asm->OutStreamer->AddComment("Argument list type index");
  Asm->EmitInt32(ArgListIdx.getIndex());

  for (MDNode *N : CU_Nodes->operands()) {
    auto *CUNode = cast<DICompileUnit>(N);
    for (auto *SP : CUNode->getSubprograms()) {
      StringRef DisplayName = SP->getDisplayName();
      Asm->OutStreamer->AddComment("Type record length");
      Asm->EmitInt16(2 + sizeof(FuncId) + DisplayName.size() + 1);
      Asm->OutStreamer->AddComment("Leaf type: LF_FUNC_ID");
      Asm->EmitInt16(LF_FUNC_ID);

      Asm->OutStreamer->AddComment("Scope type index");
      Asm->EmitInt32(TypeIndex().getIndex());
      Asm->OutStreamer->AddComment("Function type");
      Asm->EmitInt32(VoidProcIdx.getIndex());
      {
        SmallString<32> NullTerminatedString(DisplayName);
        if (NullTerminatedString.empty() || NullTerminatedString.back() != '\0')
          NullTerminatedString.push_back('\0');
        Asm->OutStreamer->AddComment("Function name");
        Asm->OutStreamer->EmitBytes(NullTerminatedString);
      }

      TypeIndex FuncIdIdx = getNextTypeIndex();
      SubprogramToFuncId.insert(std::make_pair(SP, FuncIdIdx));
    }
  }
}

void CodeViewDebug::emitInlineeLinesSubsection() {
  if (InlinedSubprograms.empty())
    return;

  MCStreamer &OS = *Asm->OutStreamer;
  MCSymbol *InlineBegin = Asm->MMI->getContext().createTempSymbol(),
           *InlineEnd = Asm->MMI->getContext().createTempSymbol();

  OS.AddComment("Inlinee lines subsection");
  OS.EmitIntValue(unsigned(ModuleSubstreamKind::InlineeLines), 4);
  OS.AddComment("Subsection size");
  OS.emitAbsoluteSymbolDiff(InlineEnd, InlineBegin, 4);
  OS.EmitLabel(InlineBegin);

  // We don't provide any extra file info.
  // FIXME: Find out if debuggers use this info.
  OS.AddComment("Inlinee lines signature");
  OS.EmitIntValue(unsigned(InlineeLinesSignature::Normal), 4);

  for (const DISubprogram *SP : InlinedSubprograms) {
    OS.AddBlankLine();
    TypeIndex TypeId = SubprogramToFuncId[SP];
    unsigned FileId = maybeRecordFile(SP->getFile());
    OS.AddComment("Inlined function " + SP->getDisplayName() + " starts at " +
                  SP->getFilename() + Twine(':') + Twine(SP->getLine()));
    OS.AddBlankLine();
    // The filechecksum table uses 8 byte entries for now, and file ids start at
    // 1.
    unsigned FileOffset = (FileId - 1) * 8;
    OS.AddComment("Type index of inlined function");
    OS.EmitIntValue(TypeId.getIndex(), 4);
    OS.AddComment("Offset into filechecksum table");
    OS.EmitIntValue(FileOffset, 4);
    OS.AddComment("Starting line number");
    OS.EmitIntValue(SP->getLine(), 4);
  }

  OS.EmitLabel(InlineEnd);
}

static void EmitLabelDiff(MCStreamer &Streamer,
                          const MCSymbol *From, const MCSymbol *To,
                          unsigned int Size = 4) {
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  MCContext &Context = Streamer.getContext();
  const MCExpr *FromRef = MCSymbolRefExpr::create(From, Variant, Context),
               *ToRef   = MCSymbolRefExpr::create(To, Variant, Context);
  const MCExpr *AddrDelta =
      MCBinaryExpr::create(MCBinaryExpr::Sub, ToRef, FromRef, Context);
  Streamer.EmitValue(AddrDelta, Size);
}

void CodeViewDebug::collectInlineSiteChildren(
    SmallVectorImpl<unsigned> &Children, const FunctionInfo &FI,
    const InlineSite &Site) {
  for (const DILocation *ChildSiteLoc : Site.ChildSites) {
    auto I = FI.InlineSites.find(ChildSiteLoc);
    assert(I != FI.InlineSites.end());
    const InlineSite &ChildSite = I->second;
    Children.push_back(ChildSite.SiteFuncId);
    collectInlineSiteChildren(Children, FI, ChildSite);
  }
}

void CodeViewDebug::emitInlinedCallSite(const FunctionInfo &FI,
                                        const DILocation *InlinedAt,
                                        const InlineSite &Site) {
  MCStreamer &OS = *Asm->OutStreamer;

  MCSymbol *InlineBegin = Asm->MMI->getContext().createTempSymbol(),
           *InlineEnd = Asm->MMI->getContext().createTempSymbol();

  assert(SubprogramToFuncId.count(Site.Inlinee));
  TypeIndex InlineeIdx = SubprogramToFuncId[Site.Inlinee];

  // SymbolRecord
  Asm->OutStreamer->AddComment("Record length");
  EmitLabelDiff(OS, InlineBegin, InlineEnd, 2);   // RecordLength
  OS.EmitLabel(InlineBegin);
  Asm->OutStreamer->AddComment("Record kind: S_INLINESITE");
  Asm->EmitInt16(SymbolRecordKind::S_INLINESITE); // RecordKind

  Asm->OutStreamer->AddComment("PtrParent");
  Asm->OutStreamer->EmitIntValue(0, 4);
  Asm->OutStreamer->AddComment("PtrEnd");
  Asm->OutStreamer->EmitIntValue(0, 4);
  Asm->OutStreamer->AddComment("Inlinee type index");
  Asm->EmitInt32(InlineeIdx.getIndex());

  unsigned FileId = maybeRecordFile(Site.Inlinee->getFile());
  unsigned StartLineNum = Site.Inlinee->getLine();
  SmallVector<unsigned, 3> SecondaryFuncIds;
  collectInlineSiteChildren(SecondaryFuncIds, FI, Site);

  OS.EmitCVInlineLinetableDirective(Site.SiteFuncId, FileId, StartLineNum,
                                    FI.Begin, FI.End, SecondaryFuncIds);

  OS.EmitLabel(InlineEnd);

  // Recurse on child inlined call sites before closing the scope.
  for (const DILocation *ChildSite : Site.ChildSites) {
    auto I = FI.InlineSites.find(ChildSite);
    assert(I != FI.InlineSites.end() &&
           "child site not in function inline site map");
    emitInlinedCallSite(FI, ChildSite, I->second);
  }

  // Close the scope.
  Asm->OutStreamer->AddComment("Record length");
  Asm->EmitInt16(2);                                  // RecordLength
  Asm->OutStreamer->AddComment("Record kind: S_INLINESITE_END");
  Asm->EmitInt16(SymbolRecordKind::S_INLINESITE_END); // RecordKind
}

void CodeViewDebug::emitDebugInfoForFunction(const Function *GV,
                                             FunctionInfo &FI) {
  // For each function there is a separate subsection
  // which holds the PC to file:line table.
  const MCSymbol *Fn = Asm->getSymbol(GV);
  assert(Fn);

  StringRef FuncName;
  if (auto *SP = getDISubprogram(GV))
    FuncName = SP->getDisplayName();

  // If our DISubprogram name is empty, use the mangled name.
  if (FuncName.empty())
    FuncName = GlobalValue::getRealLinkageName(GV->getName());

  // Emit a symbol subsection, required by VS2012+ to find function boundaries.
  MCSymbol *SymbolsBegin = Asm->MMI->getContext().createTempSymbol(),
           *SymbolsEnd = Asm->MMI->getContext().createTempSymbol();
  Asm->OutStreamer->AddComment("Symbol subsection for " + Twine(FuncName));
  Asm->EmitInt32(unsigned(ModuleSubstreamKind::Symbols));
  Asm->OutStreamer->AddComment("Subsection size");
  EmitLabelDiff(*Asm->OutStreamer, SymbolsBegin, SymbolsEnd);
  Asm->OutStreamer->EmitLabel(SymbolsBegin);
  {
    MCSymbol *ProcRecordBegin = Asm->MMI->getContext().createTempSymbol(),
             *ProcRecordEnd = Asm->MMI->getContext().createTempSymbol();
    Asm->OutStreamer->AddComment("Record length");
    EmitLabelDiff(*Asm->OutStreamer, ProcRecordBegin, ProcRecordEnd, 2);
    Asm->OutStreamer->EmitLabel(ProcRecordBegin);

    Asm->OutStreamer->AddComment("Record kind: S_GPROC32_ID");
    Asm->EmitInt16(unsigned(SymbolRecordKind::S_GPROC32_ID));

    // These fields are filled in by tools like CVPACK which run after the fact.
    Asm->OutStreamer->AddComment("PtrParent");
    Asm->OutStreamer->EmitIntValue(0, 4);
    Asm->OutStreamer->AddComment("PtrEnd");
    Asm->OutStreamer->EmitIntValue(0, 4);
    Asm->OutStreamer->AddComment("PtrNext");
    Asm->OutStreamer->EmitIntValue(0, 4);
    // This is the important bit that tells the debugger where the function
    // code is located and what's its size:
    Asm->OutStreamer->AddComment("Code size");
    EmitLabelDiff(*Asm->OutStreamer, Fn, FI.End);
    Asm->OutStreamer->AddComment("Offset after prologue");
    Asm->OutStreamer->EmitIntValue(0, 4);
    Asm->OutStreamer->AddComment("Offset before epilogue");
    Asm->OutStreamer->EmitIntValue(0, 4);
    Asm->OutStreamer->AddComment("Function type index");
    Asm->OutStreamer->EmitIntValue(0, 4);
    Asm->OutStreamer->AddComment("Function section relative address");
    Asm->OutStreamer->EmitCOFFSecRel32(Fn);
    Asm->OutStreamer->AddComment("Function section index");
    Asm->OutStreamer->EmitCOFFSectionIndex(Fn);
    Asm->OutStreamer->AddComment("Flags");
    Asm->EmitInt8(0);
    // Emit the function display name as a null-terminated string.
    Asm->OutStreamer->AddComment("Function name");
    {
      SmallString<32> NullTerminatedString(FuncName);
      if (NullTerminatedString.empty() || NullTerminatedString.back() != '\0')
        NullTerminatedString.push_back('\0');
      Asm->OutStreamer->EmitBytes(NullTerminatedString);
    }
    Asm->OutStreamer->EmitLabel(ProcRecordEnd);

    // Emit inlined call site information. Only emit functions inlined directly
    // into the parent function. We'll emit the other sites recursively as part
    // of their parent inline site.
    for (auto &KV : FI.InlineSites) {
      const DILocation *InlinedAt = KV.first;
      if (!InlinedAt->getInlinedAt())
        emitInlinedCallSite(FI, InlinedAt, KV.second);
    }

    // We're done with this function.
    Asm->OutStreamer->AddComment("Record length");
    Asm->EmitInt16(0x0002);
    Asm->OutStreamer->AddComment("Record kind: S_PROC_ID_END");
    Asm->EmitInt16(unsigned(SymbolRecordKind::S_PROC_ID_END));
  }
  Asm->OutStreamer->EmitLabel(SymbolsEnd);
  // Every subsection must be aligned to a 4-byte boundary.
  Asm->OutStreamer->EmitValueToAlignment(4);

  // We have an assembler directive that takes care of the whole line table.
  Asm->OutStreamer->EmitCVLinetableDirective(FI.FuncId, Fn, FI.End);
}

void CodeViewDebug::beginFunction(const MachineFunction *MF) {
  assert(!CurFn && "Can't process two functions at once!");

  if (!Asm || !Asm->MMI->hasDebugInfo())
    return;

  const Function *GV = MF->getFunction();
  assert(FnDebugInfo.count(GV) == false);
  CurFn = &FnDebugInfo[GV];
  CurFn->FuncId = NextFuncId++;
  CurFn->Begin = Asm->getFunctionBegin();

  // Find the end of the function prolog.
  // FIXME: is there a simpler a way to do this? Can we just search
  // for the first instruction of the function, not the last of the prolog?
  DebugLoc PrologEndLoc;
  bool EmptyPrologue = true;
  for (const auto &MBB : *MF) {
    if (PrologEndLoc)
      break;
    for (const auto &MI : MBB) {
      if (MI.isDebugValue())
        continue;

      // First known non-DBG_VALUE and non-frame setup location marks
      // the beginning of the function body.
      // FIXME: do we need the first subcondition?
      if (!MI.getFlag(MachineInstr::FrameSetup) && MI.getDebugLoc()) {
        PrologEndLoc = MI.getDebugLoc();
        break;
      }
      EmptyPrologue = false;
    }
  }
  // Record beginning of function if we have a non-empty prologue.
  if (PrologEndLoc && !EmptyPrologue) {
    DebugLoc FnStartDL = PrologEndLoc.getFnDebugLoc();
    maybeRecordLocation(FnStartDL, MF);
  }
}

void CodeViewDebug::endFunction(const MachineFunction *MF) {
  if (!Asm || !CurFn)  // We haven't created any debug info for this function.
    return;

  const Function *GV = MF->getFunction();
  assert(FnDebugInfo.count(GV));
  assert(CurFn == &FnDebugInfo[GV]);

  // Don't emit anything if we don't have any line tables.
  if (!CurFn->HaveLineInfo) {
    FnDebugInfo.erase(GV);
  } else {
    CurFn->End = Asm->getFunctionEnd();
  }
  CurFn = nullptr;
}

void CodeViewDebug::beginInstruction(const MachineInstr *MI) {
  // Ignore DBG_VALUE locations and function prologue.
  if (!Asm || MI->isDebugValue() || MI->getFlag(MachineInstr::FrameSetup))
    return;
  DebugLoc DL = MI->getDebugLoc();
  if (DL == PrevInstLoc || !DL)
    return;
  maybeRecordLocation(DL, Asm->MF);
}
}
