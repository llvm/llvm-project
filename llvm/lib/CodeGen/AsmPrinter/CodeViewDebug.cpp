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
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetFrameLowering.h"

using namespace llvm;
using namespace llvm::codeview;

CodeViewDebug::CodeViewDebug(AsmPrinter *AP)
    : DebugHandlerBase(AP), OS(*Asm->OutStreamer), CurFn(nullptr) {
  // If module doesn't have named metadata anchors or COFF debug section
  // is not available, skip any debug info related stuff.
  if (!MMI->getModule()->getNamedMetadata("llvm.dbg.cu") ||
      !AP->getObjFileLowering().getCOFFDebugSymbolsSection()) {
    Asm = nullptr;
    return;
  }

  // Tell MMI that we have debug info.
  MMI->setDebugInfoAvailability(true);
}

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
    NextId = OS.EmitCVFileDirective(NextId, FullPath);
    assert(NextId == FileIdMap.size() && ".cv_file directive failed");
  }
  return Insertion.first->second;
}

CodeViewDebug::InlineSite &
CodeViewDebug::getInlineSite(const DILocation *InlinedAt,
                             const DISubprogram *Inlinee) {
  auto Insertion = CurFn->InlineSites.insert({InlinedAt, InlineSite()});
  InlineSite *Site = &Insertion.first->second;
  if (Insertion.second) {
    Site->SiteFuncId = NextFuncId++;
    Site->Inlinee = Inlinee;
    InlinedSubprograms.insert(Inlinee);
  }
  return *Site;
}

void CodeViewDebug::recordLocalVariable(LocalVariable &&Var,
                                        const DILocation *InlinedAt) {
  if (InlinedAt) {
    // This variable was inlined. Associate it with the InlineSite.
    const DISubprogram *Inlinee = Var.DIVar->getScope()->getSubprogram();
    InlineSite &Site = getInlineSite(InlinedAt, Inlinee);
    Site.InlinedLocals.emplace_back(Var);
  } else {
    // This variable goes in the main ProcSym.
    CurFn->Locals.emplace_back(Var);
  }
}

static void addLocIfNotPresent(SmallVectorImpl<const DILocation *> &Locs,
                               const DILocation *Loc) {
  auto B = Locs.begin(), E = Locs.end();
  if (std::find(B, E, Loc) == E)
    Locs.push_back(Loc);
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
  if (const DILocation *SiteLoc = DL->getInlinedAt()) {
    const DILocation *Loc = DL.get();

    // If this location was actually inlined from somewhere else, give it the ID
    // of the inline call site.
    FuncId =
        getInlineSite(SiteLoc, Loc->getScope()->getSubprogram()).SiteFuncId;

    // Ensure we have links in the tree of inline call sites.
    bool FirstLoc = true;
    while ((SiteLoc = Loc->getInlinedAt())) {
      InlineSite &Site =
          getInlineSite(SiteLoc, Loc->getScope()->getSubprogram());
      if (!FirstLoc)
        addLocIfNotPresent(Site.ChildSites, Loc);
      FirstLoc = false;
      Loc = SiteLoc;
    }
    addLocIfNotPresent(CurFn->ChildSites, Loc);
  }

  OS.EmitCVLocDirective(FuncId, FileId, DL.getLine(), DL.getCol(),
                        /*PrologueEnd=*/false,
                        /*IsStmt=*/false, DL->getFilename());
}

void CodeViewDebug::endModule() {
  if (FnDebugInfo.empty())
    return;

  emitTypeInformation();

  // FIXME: For functions that are comdat, we should emit separate .debug$S
  // sections that are comdat associative with the main function instead of
  // having one big .debug$S section.
  assert(Asm != nullptr);
  OS.SwitchSection(Asm->getObjFileLowering().getCOFFDebugSymbolsSection());
  OS.AddComment("Debug section magic");
  OS.EmitIntValue(COFF::DEBUG_SECTION_MAGIC, 4);

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
  OS.AddComment("File index to string table offset subsection");
  OS.EmitCVFileChecksumsDirective();

  // This subsection holds the string table.
  OS.AddComment("String table");
  OS.EmitCVStringTableDirective();

  clear();
}

void CodeViewDebug::emitTypeInformation() {
  // Start the .debug$T section with 0x4.
  OS.SwitchSection(Asm->getObjFileLowering().getCOFFDebugTypesSection());
  OS.AddComment("Debug section magic");
  OS.EmitIntValue(COFF::DEBUG_SECTION_MAGIC, 4);

  NamedMDNode *CU_Nodes =
      MMI->getModule()->getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes)
    return;

  // This type info currently only holds function ids for use with inline call
  // frame info. All functions are assigned a simple 'void ()' type. Emit that
  // type here.
  TypeIndex ArgListIdx = getNextTypeIndex();
  OS.AddComment("Type record length");
  OS.EmitIntValue(2 + sizeof(ArgList), 2);
  OS.AddComment("Leaf type: LF_ARGLIST");
  OS.EmitIntValue(LF_ARGLIST, 2);
  OS.AddComment("Number of arguments");
  OS.EmitIntValue(0, 4);

  TypeIndex VoidProcIdx = getNextTypeIndex();
  OS.AddComment("Type record length");
  OS.EmitIntValue(2 + sizeof(ProcedureType), 2);
  OS.AddComment("Leaf type: LF_PROCEDURE");
  OS.EmitIntValue(LF_PROCEDURE, 2);
  OS.AddComment("Return type index");
  OS.EmitIntValue(TypeIndex::Void().getIndex(), 4);
  OS.AddComment("Calling convention");
  OS.EmitIntValue(char(CallingConvention::NearC), 1);
  OS.AddComment("Function options");
  OS.EmitIntValue(char(FunctionOptions::None), 1);
  OS.AddComment("# of parameters");
  OS.EmitIntValue(0, 2);
  OS.AddComment("Argument list type index");
  OS.EmitIntValue(ArgListIdx.getIndex(), 4);

  for (MDNode *N : CU_Nodes->operands()) {
    auto *CUNode = cast<DICompileUnit>(N);
    for (auto *SP : CUNode->getSubprograms()) {
      StringRef DisplayName = SP->getDisplayName();
      OS.AddComment("Type record length");
      OS.EmitIntValue(2 + sizeof(FuncId) + DisplayName.size() + 1, 2);
      OS.AddComment("Leaf type: LF_FUNC_ID");
      OS.EmitIntValue(LF_FUNC_ID, 2);

      OS.AddComment("Scope type index");
      OS.EmitIntValue(TypeIndex().getIndex(), 4);
      OS.AddComment("Function type");
      OS.EmitIntValue(VoidProcIdx.getIndex(), 4);
      {
        SmallString<32> NullTerminatedString(DisplayName);
        if (NullTerminatedString.empty() || NullTerminatedString.back() != '\0')
          NullTerminatedString.push_back('\0');
        OS.AddComment("Function name");
        OS.EmitBytes(NullTerminatedString);
      }

      TypeIndex FuncIdIdx = getNextTypeIndex();
      SubprogramToFuncId.insert(std::make_pair(SP, FuncIdIdx));
    }
  }
}

void CodeViewDebug::emitInlineeLinesSubsection() {
  if (InlinedSubprograms.empty())
    return;

  MCSymbol *InlineBegin = MMI->getContext().createTempSymbol(),
           *InlineEnd = MMI->getContext().createTempSymbol();

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

void CodeViewDebug::collectInlineSiteChildren(
    SmallVectorImpl<unsigned> &Children, const FunctionInfo &FI,
    const InlineSite &Site) {
  for (const DILocation *ChildSiteLoc : Site.ChildSites) {
    auto I = FI.InlineSites.find(ChildSiteLoc);
    const InlineSite &ChildSite = I->second;
    Children.push_back(ChildSite.SiteFuncId);
    collectInlineSiteChildren(Children, FI, ChildSite);
  }
}

void CodeViewDebug::emitInlinedCallSite(const FunctionInfo &FI,
                                        const DILocation *InlinedAt,
                                        const InlineSite &Site) {
  MCSymbol *InlineBegin = MMI->getContext().createTempSymbol(),
           *InlineEnd = MMI->getContext().createTempSymbol();

  assert(SubprogramToFuncId.count(Site.Inlinee));
  TypeIndex InlineeIdx = SubprogramToFuncId[Site.Inlinee];

  // SymbolRecord
  OS.AddComment("Record length");
  OS.emitAbsoluteSymbolDiff(InlineEnd, InlineBegin, 2);   // RecordLength
  OS.EmitLabel(InlineBegin);
  OS.AddComment("Record kind: S_INLINESITE");
  OS.EmitIntValue(SymbolRecordKind::S_INLINESITE, 2); // RecordKind

  OS.AddComment("PtrParent");
  OS.EmitIntValue(0, 4);
  OS.AddComment("PtrEnd");
  OS.EmitIntValue(0, 4);
  OS.AddComment("Inlinee type index");
  OS.EmitIntValue(InlineeIdx.getIndex(), 4);

  unsigned FileId = maybeRecordFile(Site.Inlinee->getFile());
  unsigned StartLineNum = Site.Inlinee->getLine();
  SmallVector<unsigned, 3> SecondaryFuncIds;
  collectInlineSiteChildren(SecondaryFuncIds, FI, Site);

  OS.EmitCVInlineLinetableDirective(Site.SiteFuncId, FileId, StartLineNum,
                                    FI.Begin, FI.End, SecondaryFuncIds);

  OS.EmitLabel(InlineEnd);

  for (const LocalVariable &Var : Site.InlinedLocals)
    emitLocalVariable(Var);

  // Recurse on child inlined call sites before closing the scope.
  for (const DILocation *ChildSite : Site.ChildSites) {
    auto I = FI.InlineSites.find(ChildSite);
    assert(I != FI.InlineSites.end() &&
           "child site not in function inline site map");
    emitInlinedCallSite(FI, ChildSite, I->second);
  }

  // Close the scope.
  OS.AddComment("Record length");
  OS.EmitIntValue(2, 2);                                  // RecordLength
  OS.AddComment("Record kind: S_INLINESITE_END");
  OS.EmitIntValue(SymbolRecordKind::S_INLINESITE_END, 2); // RecordKind
}

static void emitNullTerminatedString(MCStreamer &OS, StringRef S) {
  SmallString<32> NullTerminatedString(S);
  if (NullTerminatedString.empty() || NullTerminatedString.back() != '\0')
    NullTerminatedString.push_back('\0');
  OS.EmitBytes(NullTerminatedString);
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
  MCSymbol *SymbolsBegin = MMI->getContext().createTempSymbol(),
           *SymbolsEnd = MMI->getContext().createTempSymbol();
  OS.AddComment("Symbol subsection for " + Twine(FuncName));
  OS.EmitIntValue(unsigned(ModuleSubstreamKind::Symbols), 4);
  OS.AddComment("Subsection size");
  OS.emitAbsoluteSymbolDiff(SymbolsEnd, SymbolsBegin, 4);
  OS.EmitLabel(SymbolsBegin);
  {
    MCSymbol *ProcRecordBegin = MMI->getContext().createTempSymbol(),
             *ProcRecordEnd = MMI->getContext().createTempSymbol();
    OS.AddComment("Record length");
    OS.emitAbsoluteSymbolDiff(ProcRecordEnd, ProcRecordBegin, 2);
    OS.EmitLabel(ProcRecordBegin);

    OS.AddComment("Record kind: S_GPROC32_ID");
    OS.EmitIntValue(unsigned(SymbolRecordKind::S_GPROC32_ID), 2);

    // These fields are filled in by tools like CVPACK which run after the fact.
    OS.AddComment("PtrParent");
    OS.EmitIntValue(0, 4);
    OS.AddComment("PtrEnd");
    OS.EmitIntValue(0, 4);
    OS.AddComment("PtrNext");
    OS.EmitIntValue(0, 4);
    // This is the important bit that tells the debugger where the function
    // code is located and what's its size:
    OS.AddComment("Code size");
    OS.emitAbsoluteSymbolDiff(FI.End, Fn, 4);
    OS.AddComment("Offset after prologue");
    OS.EmitIntValue(0, 4);
    OS.AddComment("Offset before epilogue");
    OS.EmitIntValue(0, 4);
    OS.AddComment("Function type index");
    OS.EmitIntValue(0, 4);
    OS.AddComment("Function section relative address");
    OS.EmitCOFFSecRel32(Fn);
    OS.AddComment("Function section index");
    OS.EmitCOFFSectionIndex(Fn);
    OS.AddComment("Flags");
    OS.EmitIntValue(0, 1);
    // Emit the function display name as a null-terminated string.
    OS.AddComment("Function name");
    emitNullTerminatedString(OS, FuncName);
    OS.EmitLabel(ProcRecordEnd);

    for (const LocalVariable &Var : FI.Locals)
      emitLocalVariable(Var);

    // Emit inlined call site information. Only emit functions inlined directly
    // into the parent function. We'll emit the other sites recursively as part
    // of their parent inline site.
    for (const DILocation *InlinedAt : FI.ChildSites) {
      auto I = FI.InlineSites.find(InlinedAt);
      assert(I != FI.InlineSites.end() &&
             "child site not in function inline site map");
      emitInlinedCallSite(FI, InlinedAt, I->second);
    }

    // We're done with this function.
    OS.AddComment("Record length");
    OS.EmitIntValue(0x0002, 2);
    OS.AddComment("Record kind: S_PROC_ID_END");
    OS.EmitIntValue(unsigned(SymbolRecordKind::S_PROC_ID_END), 2);
  }
  OS.EmitLabel(SymbolsEnd);
  // Every subsection must be aligned to a 4-byte boundary.
  OS.EmitValueToAlignment(4);

  // We have an assembler directive that takes care of the whole line table.
  OS.EmitCVLinetableDirective(FI.FuncId, Fn, FI.End);
}

CodeViewDebug::LocalVarDefRange
CodeViewDebug::createDefRangeMem(uint16_t CVRegister, int Offset) {
  LocalVarDefRange DR;
  DR.InMemory = 1;
  DR.DataOffset = Offset;
  assert(DR.DataOffset == Offset && "truncation");
  DR.StructOffset = 0;
  DR.CVRegister = CVRegister;
  return DR;
}

CodeViewDebug::LocalVarDefRange
CodeViewDebug::createDefRangeReg(uint16_t CVRegister) {
  LocalVarDefRange DR;
  DR.InMemory = 0;
  DR.DataOffset = 0;
  DR.StructOffset = 0;
  DR.CVRegister = CVRegister;
  return DR;
}

void CodeViewDebug::collectVariableInfoFromMMITable(
    DenseSet<InlinedVariable> &Processed) {
  const TargetSubtargetInfo &TSI = Asm->MF->getSubtarget();
  const TargetFrameLowering *TFI = TSI.getFrameLowering();
  const TargetRegisterInfo *TRI = TSI.getRegisterInfo();

  for (const MachineModuleInfo::VariableDbgInfo &VI :
       MMI->getVariableDbgInfo()) {
    if (!VI.Var)
      continue;
    assert(VI.Var->isValidLocationForIntrinsic(VI.Loc) &&
           "Expected inlined-at fields to agree");

    Processed.insert(InlinedVariable(VI.Var, VI.Loc->getInlinedAt()));
    LexicalScope *Scope = LScopes.findLexicalScope(VI.Loc);

    // If variable scope is not found then skip this variable.
    if (!Scope)
      continue;

    // Get the frame register used and the offset.
    unsigned FrameReg = 0;
    int FrameOffset = TFI->getFrameIndexReference(*Asm->MF, VI.Slot, FrameReg);
    uint16_t CVReg = TRI->getCodeViewRegNum(FrameReg);

    // Calculate the label ranges.
    LocalVarDefRange DefRange = createDefRangeMem(CVReg, FrameOffset);
    for (const InsnRange &Range : Scope->getRanges()) {
      const MCSymbol *Begin = getLabelBeforeInsn(Range.first);
      const MCSymbol *End = getLabelAfterInsn(Range.second);
      End = End ? End : Asm->getFunctionEnd();
      DefRange.Ranges.emplace_back(Begin, End);
    }

    LocalVariable Var;
    Var.DIVar = VI.Var;
    Var.DefRanges.emplace_back(std::move(DefRange));
    recordLocalVariable(std::move(Var), VI.Loc->getInlinedAt());
  }
}

void CodeViewDebug::collectVariableInfo(const DISubprogram *SP) {
  DenseSet<InlinedVariable> Processed;
  // Grab the variable info that was squirreled away in the MMI side-table.
  collectVariableInfoFromMMITable(Processed);

  const TargetRegisterInfo *TRI = Asm->MF->getSubtarget().getRegisterInfo();

  for (const auto &I : DbgValues) {
    InlinedVariable IV = I.first;
    if (Processed.count(IV))
      continue;
    const DILocalVariable *DIVar = IV.first;
    const DILocation *InlinedAt = IV.second;

    // Instruction ranges, specifying where IV is accessible.
    const auto &Ranges = I.second;

    LexicalScope *Scope = nullptr;
    if (InlinedAt)
      Scope = LScopes.findInlinedScope(DIVar->getScope(), InlinedAt);
    else
      Scope = LScopes.findLexicalScope(DIVar->getScope());
    // If variable scope is not found then skip this variable.
    if (!Scope)
      continue;

    LocalVariable Var;
    Var.DIVar = DIVar;

    // Calculate the definition ranges.
    for (auto I = Ranges.begin(), E = Ranges.end(); I != E; ++I) {
      const InsnRange &Range = *I;
      const MachineInstr *DVInst = Range.first;
      assert(DVInst->isDebugValue() && "Invalid History entry");
      const DIExpression *DIExpr = DVInst->getDebugExpression();

      // Bail if there is a complex DWARF expression for now.
      if (DIExpr && DIExpr->getNumElements() > 0)
        continue;

      // Handle the two cases we can handle: indirect in memory and in register.
      bool IsIndirect = DVInst->getOperand(1).isImm();
      unsigned CVReg = TRI->getCodeViewRegNum(DVInst->getOperand(0).getReg());
      {
        LocalVarDefRange DefRange;
        if (IsIndirect) {
          int64_t Offset = DVInst->getOperand(1).getImm();
          DefRange = createDefRangeMem(CVReg, Offset);
        } else {
          DefRange = createDefRangeReg(CVReg);
        }
        if (Var.DefRanges.empty() ||
            Var.DefRanges.back().isDifferentLocation(DefRange)) {
          Var.DefRanges.emplace_back(std::move(DefRange));
        }
      }

      // Compute the label range.
      const MCSymbol *Begin = getLabelBeforeInsn(Range.first);
      const MCSymbol *End = getLabelAfterInsn(Range.second);
      if (!End) {
        if (std::next(I) != E)
          End = getLabelBeforeInsn(std::next(I)->first);
        else
          End = Asm->getFunctionEnd();
      }

      // If the last range end is our begin, just extend the last range.
      // Otherwise make a new range.
      SmallVectorImpl<std::pair<const MCSymbol *, const MCSymbol *>> &Ranges =
          Var.DefRanges.back().Ranges;
      if (!Ranges.empty() && Ranges.back().second == Begin)
        Ranges.back().second = End;
      else
        Ranges.emplace_back(Begin, End);

      // FIXME: Do more range combining.
    }

    recordLocalVariable(std::move(Var), InlinedAt);
  }
}

void CodeViewDebug::beginFunction(const MachineFunction *MF) {
  assert(!CurFn && "Can't process two functions at once!");

  if (!Asm || !MMI->hasDebugInfo())
    return;

  DebugHandlerBase::beginFunction(MF);

  const Function *GV = MF->getFunction();
  assert(FnDebugInfo.count(GV) == false);
  CurFn = &FnDebugInfo[GV];
  CurFn->FuncId = NextFuncId++;
  CurFn->Begin = Asm->getFunctionBegin();

  // Find the end of the function prolog.  First known non-DBG_VALUE and
  // non-frame setup location marks the beginning of the function body.
  // FIXME: is there a simpler a way to do this? Can we just search
  // for the first instruction of the function, not the last of the prolog?
  DebugLoc PrologEndLoc;
  bool EmptyPrologue = true;
  for (const auto &MBB : *MF) {
    for (const auto &MI : MBB) {
      if (!MI.isDebugValue() && !MI.getFlag(MachineInstr::FrameSetup) &&
          MI.getDebugLoc()) {
        PrologEndLoc = MI.getDebugLoc();
        break;
      } else if (!MI.isDebugValue()) {
        EmptyPrologue = false;
      }
    }
  }

  // Record beginning of function if we have a non-empty prologue.
  if (PrologEndLoc && !EmptyPrologue) {
    DebugLoc FnStartDL = PrologEndLoc.getFnDebugLoc();
    maybeRecordLocation(FnStartDL, MF);
  }
}

void CodeViewDebug::emitLocalVariable(const LocalVariable &Var) {
  // LocalSym record, see SymbolRecord.h for more info.
  MCSymbol *LocalBegin = MMI->getContext().createTempSymbol(),
           *LocalEnd = MMI->getContext().createTempSymbol();
  OS.AddComment("Record length");
  OS.emitAbsoluteSymbolDiff(LocalEnd, LocalBegin, 2);
  OS.EmitLabel(LocalBegin);

  OS.AddComment("Record kind: S_LOCAL");
  OS.EmitIntValue(unsigned(SymbolRecordKind::S_LOCAL), 2);

  uint16_t Flags = 0;
  if (Var.DIVar->isParameter())
    Flags |= LocalSym::IsParameter;
  if (Var.DefRanges.empty())
    Flags |= LocalSym::IsOptimizedOut;

  OS.AddComment("TypeIndex");
  OS.EmitIntValue(TypeIndex::Int32().getIndex(), 4);
  OS.AddComment("Flags");
  OS.EmitIntValue(Flags, 2);
  emitNullTerminatedString(OS, Var.DIVar->getName());
  OS.EmitLabel(LocalEnd);

  // Calculate the on disk prefix of the appropriate def range record. The
  // records and on disk formats are described in SymbolRecords.h. BytePrefix
  // should be big enough to hold all forms without memory allocation.
  SmallString<20> BytePrefix;
  for (const LocalVarDefRange &DefRange : Var.DefRanges) {
    BytePrefix.clear();
    // FIXME: Handle bitpieces.
    if (DefRange.StructOffset != 0)
      continue;

    if (DefRange.InMemory) {
      DefRangeRegisterRelSym Sym{};
      ulittle16_t SymKind = ulittle16_t(S_DEFRANGE_REGISTER_REL);
      Sym.BaseRegister = DefRange.CVRegister;
      Sym.Flags = 0; // Unclear what matters here.
      Sym.BasePointerOffset = DefRange.DataOffset;
      BytePrefix +=
          StringRef(reinterpret_cast<const char *>(&SymKind), sizeof(SymKind));
      BytePrefix += StringRef(reinterpret_cast<const char *>(&Sym),
                              sizeof(Sym) - sizeof(LocalVariableAddrRange));
    } else {
      assert(DefRange.DataOffset == 0 && "unexpected offset into register");
      DefRangeRegisterSym Sym{};
      ulittle16_t SymKind = ulittle16_t(S_DEFRANGE_REGISTER);
      Sym.Register = DefRange.CVRegister;
      Sym.MayHaveNoName = 0; // Unclear what matters here.
      BytePrefix +=
          StringRef(reinterpret_cast<const char *>(&SymKind), sizeof(SymKind));
      BytePrefix += StringRef(reinterpret_cast<const char *>(&Sym),
                              sizeof(Sym) - sizeof(LocalVariableAddrRange));
    }
    OS.EmitCVDefRangeDirective(DefRange.Ranges, BytePrefix);
  }
}

void CodeViewDebug::endFunction(const MachineFunction *MF) {
  if (!Asm || !CurFn)  // We haven't created any debug info for this function.
    return;

  const Function *GV = MF->getFunction();
  assert(FnDebugInfo.count(GV));
  assert(CurFn == &FnDebugInfo[GV]);

  collectVariableInfo(getDISubprogram(GV));

  DebugHandlerBase::endFunction(MF);

  // Don't emit anything if we don't have any line tables.
  if (!CurFn->HaveLineInfo) {
    FnDebugInfo.erase(GV);
    CurFn = nullptr;
    return;
  }

  CurFn->End = Asm->getFunctionEnd();

  CurFn = nullptr;
}

void CodeViewDebug::beginInstruction(const MachineInstr *MI) {
  DebugHandlerBase::beginInstruction(MI);

  // Ignore DBG_VALUE locations and function prologue.
  if (!Asm || MI->isDebugValue() || MI->getFlag(MachineInstr::FrameSetup))
    return;
  DebugLoc DL = MI->getDebugLoc();
  if (DL == PrevInstLoc || !DL)
    return;
  maybeRecordLocation(DL, Asm->MF);
}
