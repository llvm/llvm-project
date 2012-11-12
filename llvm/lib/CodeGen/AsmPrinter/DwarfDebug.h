//===-- llvm/CodeGen/DwarfDebug.h - Dwarf Debug Framework ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFDEBUG_H__
#define CODEGEN_ASMPRINTER_DWARFDEBUG_H__

#include "DIE.h"
#include "llvm/DebugInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/DebugLoc.h"

namespace llvm {

class CompileUnit;
class ConstantInt;
class ConstantFP;
class DbgVariable;
class MachineFrameInfo;
class MachineModuleInfo;
class MachineOperand;
class MCAsmInfo;
class DIEAbbrev;
class DIE;
class DIEBlock;
class DIEEntry;

//===----------------------------------------------------------------------===//
/// SrcLineInfo - This class is used to record source line correspondence.
///
class SrcLineInfo {
  unsigned Line;                     // Source line number.
  unsigned Column;                   // Source column.
  unsigned SourceID;                 // Source ID number.
  MCSymbol *Label;                   // Label in code ID number.
public:
  SrcLineInfo(unsigned L, unsigned C, unsigned S, MCSymbol *label)
    : Line(L), Column(C), SourceID(S), Label(label) {}

  // Accessors
  unsigned getLine() const { return Line; }
  unsigned getColumn() const { return Column; }
  unsigned getSourceID() const { return SourceID; }
  MCSymbol *getLabel() const { return Label; }
};

/// DotDebugLocEntry - This struct describes location entries emitted in
/// .debug_loc section.
typedef struct DotDebugLocEntry {
  const MCSymbol *Begin;
  const MCSymbol *End;
  MachineLocation Loc;
  const MDNode *Variable;
  bool Merged;
  bool Constant;
  enum EntryType {
    E_Location,
    E_Integer,
    E_ConstantFP,
    E_ConstantInt
  };
  enum EntryType EntryKind;

  union {
    int64_t Int;
    const ConstantFP *CFP;
    const ConstantInt *CIP;
  } Constants;
  DotDebugLocEntry() 
    : Begin(0), End(0), Variable(0), Merged(false), 
      Constant(false) { Constants.Int = 0;}
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, MachineLocation &L,
                   const MDNode *V) 
    : Begin(B), End(E), Loc(L), Variable(V), Merged(false), 
      Constant(false) { Constants.Int = 0; EntryKind = E_Location; }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, int64_t i)
    : Begin(B), End(E), Variable(0), Merged(false), 
      Constant(true) { Constants.Int = i; EntryKind = E_Integer; }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E, const ConstantFP *FPtr)
    : Begin(B), End(E), Variable(0), Merged(false), 
      Constant(true) { Constants.CFP = FPtr; EntryKind = E_ConstantFP; }
  DotDebugLocEntry(const MCSymbol *B, const MCSymbol *E,
                   const ConstantInt *IPtr)
    : Begin(B), End(E), Variable(0), Merged(false), 
      Constant(true) { Constants.CIP = IPtr; EntryKind = E_ConstantInt; }

  /// Empty entries are also used as a trigger to emit temp label. Such
  /// labels are referenced is used to find debug_loc offset for a given DIE.
  bool isEmpty() { return Begin == 0 && End == 0; }
  bool isMerged() { return Merged; }
  void Merge(DotDebugLocEntry *Next) {
    if (!(Begin && Loc == Next->Loc && End == Next->Begin))
      return;
    Next->Begin = Begin;
    Merged = true;
  }
  bool isLocation() const    { return EntryKind == E_Location; }
  bool isInt() const         { return EntryKind == E_Integer; }
  bool isConstantFP() const  { return EntryKind == E_ConstantFP; }
  bool isConstantInt() const { return EntryKind == E_ConstantInt; }
  int64_t getInt()                    { return Constants.Int; }
  const ConstantFP *getConstantFP()   { return Constants.CFP; }
  const ConstantInt *getConstantInt() { return Constants.CIP; }
} DotDebugLocEntry;

//===----------------------------------------------------------------------===//
/// DbgVariable - This class is used to track local variable information.
///
class DbgVariable {
  DIVariable Var;                    // Variable Descriptor.
  DIE *TheDIE;                       // Variable DIE.
  unsigned DotDebugLocOffset;        // Offset in DotDebugLocEntries.
  DbgVariable *AbsVar;               // Corresponding Abstract variable, if any.
  const MachineInstr *MInsn;         // DBG_VALUE instruction of the variable.
  int FrameIndex;
public:
  // AbsVar may be NULL.
  DbgVariable(DIVariable V, DbgVariable *AV) 
    : Var(V), TheDIE(0), DotDebugLocOffset(~0U), AbsVar(AV), MInsn(0),
      FrameIndex(~0) {}

  // Accessors.
  DIVariable getVariable()           const { return Var; }
  void setDIE(DIE *D)                      { TheDIE = D; }
  DIE *getDIE()                      const { return TheDIE; }
  void setDotDebugLocOffset(unsigned O)    { DotDebugLocOffset = O; }
  unsigned getDotDebugLocOffset()    const { return DotDebugLocOffset; }
  StringRef getName()                const { return Var.getName(); }
  DbgVariable *getAbstractVariable() const { return AbsVar; }
  const MachineInstr *getMInsn()     const { return MInsn; }
  void setMInsn(const MachineInstr *M)     { MInsn = M; }
  int getFrameIndex()                const { return FrameIndex; }
  void setFrameIndex(int FI)               { FrameIndex = FI; }
  // Translate tag to proper Dwarf tag.  
  unsigned getTag()                  const { 
    if (Var.getTag() == dwarf::DW_TAG_arg_variable)
      return dwarf::DW_TAG_formal_parameter;
    
    return dwarf::DW_TAG_variable;
  }
  /// isArtificial - Return true if DbgVariable is artificial.
  bool isArtificial()                const {
    if (Var.isArtificial())
      return true;
    if (getType().isArtificial())
      return true;
    return false;
  }

  bool isObjectPointer()             const {
    if (Var.isObjectPointer())
      return true;
    if (getType().isObjectPointer())
      return true;
    return false;
  }
  
  bool variableHasComplexAddress()   const {
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.hasComplexAddress();
  }
  bool isBlockByrefVariable()        const {
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.isBlockByrefVariable();
  }
  unsigned getNumAddrElements()      const { 
    assert(Var.Verify() && "Invalid complex DbgVariable!");
    return Var.getNumAddrElements();
  }
  uint64_t getAddrElement(unsigned i) const {
    return Var.getAddrElement(i);
  }
  DIType getType() const;
};

class DwarfDebug {
  /// Asm - Target of Dwarf emission.
  AsmPrinter *Asm;

  /// MMI - Collected machine module information.
  MachineModuleInfo *MMI;

  /// DIEValueAllocator - All DIEValues are allocated through this allocator.
  BumpPtrAllocator DIEValueAllocator;

  //===--------------------------------------------------------------------===//
  // Attributes used to construct specific Dwarf sections.
  //

  CompileUnit *FirstCU;

  /// Maps MDNode with its corresponding CompileUnit.
  DenseMap <const MDNode *, CompileUnit *> CUMap;

  /// Maps subprogram MDNode with its corresponding CompileUnit.
  DenseMap <const MDNode *, CompileUnit *> SPMap;

  /// AbbreviationsSet - Used to uniquely define abbreviations.
  ///
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  /// Abbreviations - A list of all the unique abbreviations in use.
  ///
  std::vector<DIEAbbrev *> Abbreviations;

  /// SourceIdMap - Source id map, i.e. pair of source filename and directory,
  /// separated by a zero byte, mapped to a unique id.
  StringMap<unsigned, BumpPtrAllocator&> SourceIdMap;

  /// StringPool - A String->Symbol mapping of strings used by indirect
  /// references.
  StringMap<std::pair<MCSymbol*, unsigned>, BumpPtrAllocator&> StringPool;
  unsigned NextStringPoolNumber;
  
  /// SectionMap - Provides a unique id per text section.
  ///
  SetVector<const MCSection*> SectionMap;

  /// CurrentFnArguments - List of Arguments (DbgValues) for current function.
  SmallVector<DbgVariable *, 8> CurrentFnArguments;

  LexicalScopes LScopes;

  /// AbstractSPDies - Collection of abstract subprogram DIEs.
  DenseMap<const MDNode *, DIE *> AbstractSPDies;

  /// ScopeVariables - Collection of dbg variables of a scope.
  DenseMap<LexicalScope *, SmallVector<DbgVariable *, 8> > ScopeVariables;

  /// AbstractVariables - Collection of abstract variables.
  DenseMap<const MDNode *, DbgVariable *> AbstractVariables;

  /// DotDebugLocEntries - Collection of DotDebugLocEntry.
  SmallVector<DotDebugLocEntry, 4> DotDebugLocEntries;

  /// InlinedSubprogramDIEs - Collection of subprogram DIEs that are marked
  /// (at the end of the module) as DW_AT_inline.
  SmallPtrSet<DIE *, 4> InlinedSubprogramDIEs;

  /// InlineInfo - Keep track of inlined functions and their location.  This
  /// information is used to populate the debug_inlined section.
  typedef std::pair<const MCSymbol *, DIE *> InlineInfoLabels;
  DenseMap<const MDNode *, SmallVector<InlineInfoLabels, 4> > InlineInfo;
  SmallVector<const MDNode *, 4> InlinedSPNodes;

  // ProcessedSPNodes - This is a collection of subprogram MDNodes that
  // are processed to create DIEs.
  SmallPtrSet<const MDNode *, 16> ProcessedSPNodes;

  /// LabelsBeforeInsn - Maps instruction with label emitted before 
  /// instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsBeforeInsn;

  /// LabelsAfterInsn - Maps instruction with label emitted after
  /// instruction.
  DenseMap<const MachineInstr *, MCSymbol *> LabelsAfterInsn;

  /// UserVariables - Every user variable mentioned by a DBG_VALUE instruction
  /// in order of appearance.
  SmallVector<const MDNode*, 8> UserVariables;

  /// DbgValues - For each user variable, keep a list of DBG_VALUE
  /// instructions in order. The list can also contain normal instructions that
  /// clobber the previous DBG_VALUE.
  typedef DenseMap<const MDNode*, SmallVector<const MachineInstr*, 4> >
    DbgValueHistoryMap;
  DbgValueHistoryMap DbgValues;

  SmallVector<const MCSymbol *, 8> DebugRangeSymbols;

  /// Previous instruction's location information. This is used to determine
  /// label location to indicate scope boundries in dwarf debug info.
  DebugLoc PrevInstLoc;
  MCSymbol *PrevLabel;

  /// PrologEndLoc - This location indicates end of function prologue and
  /// beginning of function body.
  DebugLoc PrologEndLoc;

  struct FunctionDebugFrameInfo {
    unsigned Number;
    std::vector<MachineMove> Moves;

    FunctionDebugFrameInfo(unsigned Num, const std::vector<MachineMove> &M)
      : Number(Num), Moves(M) {}
  };

  std::vector<FunctionDebugFrameInfo> DebugFrames;

  // Section Symbols: these are assembler temporary labels that are emitted at
  // the beginning of each supported dwarf section.  These are used to form
  // section offsets and are created by EmitSectionLabels.
  MCSymbol *DwarfInfoSectionSym, *DwarfAbbrevSectionSym;
  MCSymbol *DwarfStrSectionSym, *TextSectionSym, *DwarfDebugRangeSectionSym;
  MCSymbol *DwarfDebugLocSectionSym;
  MCSymbol *FunctionBeginSym, *FunctionEndSym;

  // As an optimization, there is no need to emit an entry in the directory
  // table for the same directory as DW_at_comp_dir.
  StringRef CompilationDir;

  // A holder for the DarwinGDBCompat flag so that the compile unit can use it.
  bool isDarwinGDBCompat;
  bool hasDwarfAccelTables;
private:

  /// assignAbbrevNumber - Define a unique number for the abbreviation.
  ///
  void assignAbbrevNumber(DIEAbbrev &Abbrev);

  void addScopeVariable(LexicalScope *LS, DbgVariable *Var);

  /// findAbstractVariable - Find abstract variable associated with Var.
  DbgVariable *findAbstractVariable(DIVariable &Var, DebugLoc Loc);

  /// updateSubprogramScopeDIE - Find DIE for the given subprogram and 
  /// attach appropriate DW_AT_low_pc and DW_AT_high_pc attributes.
  /// If there are global variables in this scope then create and insert
  /// DIEs for these variables.
  DIE *updateSubprogramScopeDIE(CompileUnit *SPCU, const MDNode *SPNode);

  /// constructLexicalScope - Construct new DW_TAG_lexical_block 
  /// for this scope and attach DW_AT_low_pc/DW_AT_high_pc labels.
  DIE *constructLexicalScopeDIE(CompileUnit *TheCU, LexicalScope *Scope);

  /// constructInlinedScopeDIE - This scope represents inlined body of
  /// a function. Construct DIE to represent this concrete inlined copy
  /// of the function.
  DIE *constructInlinedScopeDIE(CompileUnit *TheCU, LexicalScope *Scope);

  /// constructScopeDIE - Construct a DIE for this scope.
  DIE *constructScopeDIE(CompileUnit *TheCU, LexicalScope *Scope);

  /// EmitSectionLabels - Emit initial Dwarf sections with a label at
  /// the start of each one.
  void EmitSectionLabels();

  /// emitDIE - Recursively Emits a debug information entry.
  ///
  void emitDIE(DIE *Die);

  /// computeSizeAndOffset - Compute the size and offset of a DIE.
  ///
  unsigned computeSizeAndOffset(DIE *Die, unsigned Offset, bool Last);

  /// computeSizeAndOffsets - Compute the size and offset of all the DIEs.
  ///
  void computeSizeAndOffsets();

  /// EmitDebugInfo - Emit the debug info section.
  ///
  void emitDebugInfo();

  /// emitAbbreviations - Emit the abbreviation section.
  ///
  void emitAbbreviations() const;

  /// emitEndOfLineMatrix - Emit the last address of the section and the end of
  /// the line matrix.
  ///
  void emitEndOfLineMatrix(unsigned SectionEnd);

  /// emitAccelNames - Emit visible names into a hashed accelerator table
  /// section.
  void emitAccelNames();
  
  /// emitAccelObjC - Emit objective C classes and categories into a hashed
  /// accelerator table section.
  void emitAccelObjC();

  /// emitAccelNamespace - Emit namespace dies into a hashed accelerator
  /// table.
  void emitAccelNamespaces();

  /// emitAccelTypes() - Emit type dies into a hashed accelerator table.
  ///
  void emitAccelTypes();
  
  /// emitDebugPubTypes - Emit visible types into a debug pubtypes section.
  ///
  void emitDebugPubTypes();

  /// emitDebugStr - Emit visible names into a debug str section.
  ///
  void emitDebugStr();

  /// emitDebugLoc - Emit visible names into a debug loc section.
  ///
  void emitDebugLoc();

  /// EmitDebugARanges - Emit visible names into a debug aranges section.
  ///
  void EmitDebugARanges();

  /// emitDebugRanges - Emit visible names into a debug ranges section.
  ///
  void emitDebugRanges();

  /// emitDebugMacInfo - Emit visible names into a debug macinfo section.
  ///
  void emitDebugMacInfo();

  /// emitDebugInlineInfo - Emit inline info using following format.
  /// Section Header:
  /// 1. length of section
  /// 2. Dwarf version number
  /// 3. address size.
  ///
  /// Entries (one "entry" for each function that was inlined):
  ///
  /// 1. offset into __debug_str section for MIPS linkage name, if exists; 
  ///   otherwise offset into __debug_str for regular function name.
  /// 2. offset into __debug_str section for regular function name.
  /// 3. an unsigned LEB128 number indicating the number of distinct inlining 
  /// instances for the function.
  /// 
  /// The rest of the entry consists of a {die_offset, low_pc} pair for each 
  /// inlined instance; the die_offset points to the inlined_subroutine die in
  /// the __debug_info section, and the low_pc is the starting address for the
  /// inlining instance.
  void emitDebugInlineInfo();

  /// constructCompileUnit - Create new CompileUnit for the given 
  /// metadata node with tag DW_TAG_compile_unit.
  CompileUnit *constructCompileUnit(const MDNode *N);

  /// construct SubprogramDIE - Construct subprogram DIE.
  void constructSubprogramDIE(CompileUnit *TheCU, const MDNode *N);

  /// recordSourceLine - Register a source line with debug info. Returns the
  /// unique label that was emitted and which provides correspondence to
  /// the source line list.
  void recordSourceLine(unsigned Line, unsigned Col, const MDNode *Scope,
                        unsigned Flags);
  
  /// identifyScopeMarkers() - Indentify instructions that are marking the
  /// beginning of or ending of a scope.
  void identifyScopeMarkers();

  /// addCurrentFnArgument - If Var is an current function argument that add
  /// it in CurrentFnArguments list.
  bool addCurrentFnArgument(const MachineFunction *MF,
                            DbgVariable *Var, LexicalScope *Scope);

  /// collectVariableInfo - Populate LexicalScope entries with variables' info.
  void collectVariableInfo(const MachineFunction *,
                           SmallPtrSet<const MDNode *, 16> &ProcessedVars);
  
  /// collectVariableInfoFromMMITable - Collect variable information from
  /// side table maintained by MMI.
  void collectVariableInfoFromMMITable(const MachineFunction * MF,
                                       SmallPtrSet<const MDNode *, 16> &P);

  /// requestLabelBeforeInsn - Ensure that a label will be emitted before MI.
  void requestLabelBeforeInsn(const MachineInstr *MI) {
    LabelsBeforeInsn.insert(std::make_pair(MI, (MCSymbol*)0));
  }

  /// getLabelBeforeInsn - Return Label preceding the instruction.
  const MCSymbol *getLabelBeforeInsn(const MachineInstr *MI);

  /// requestLabelAfterInsn - Ensure that a label will be emitted after MI.
  void requestLabelAfterInsn(const MachineInstr *MI) {
    LabelsAfterInsn.insert(std::make_pair(MI, (MCSymbol*)0));
  }

  /// getLabelAfterInsn - Return Label immediately following the instruction.
  const MCSymbol *getLabelAfterInsn(const MachineInstr *MI);

public:
  //===--------------------------------------------------------------------===//
  // Main entry points.
  //
  DwarfDebug(AsmPrinter *A, Module *M);
  ~DwarfDebug();

  /// collectInfoFromNamedMDNodes - Collect debug info from named mdnodes such
  /// as llvm.dbg.enum and llvm.dbg.ty
  void collectInfoFromNamedMDNodes(Module *M);

  /// collectLegacyDebugInfo - Collect debug info using DebugInfoFinder.
  /// FIXME - Remove this when DragonEgg switches to DIBuilder.
  bool collectLegacyDebugInfo(Module *M);

  /// beginModule - Emit all Dwarf sections that should come prior to the
  /// content.
  void beginModule(Module *M);

  /// endModule - Emit all Dwarf sections that should come after the content.
  ///
  void endModule();

  /// beginFunction - Gather pre-function debug information.  Assumes being
  /// emitted immediately after the function entry point.
  void beginFunction(const MachineFunction *MF);

  /// endFunction - Gather and emit post-function debug information.
  ///
  void endFunction(const MachineFunction *MF);

  /// beginInstruction - Process beginning of an instruction.
  void beginInstruction(const MachineInstr *MI);

  /// endInstruction - Prcess end of an instruction.
  void endInstruction(const MachineInstr *MI);

  /// GetOrCreateSourceID - Look up the source id with the given directory and
  /// source file names. If none currently exists, create a new id and insert it
  /// in the SourceIds map.
  unsigned GetOrCreateSourceID(StringRef DirName, StringRef FullName);

  /// getStringPool - returns the entry into the start of the pool.
  MCSymbol *getStringPool();

  /// getStringPoolEntry - returns an entry into the string pool with the given
  /// string text.
  MCSymbol *getStringPoolEntry(StringRef Str);

  /// useDarwinGDBCompat - returns whether or not to limit some of our debug
  /// output to the limitations of darwin gdb.
  bool useDarwinGDBCompat() { return isDarwinGDBCompat; }
  bool useDwarfAccelTables() { return hasDwarfAccelTables; }
};
} // End of namespace llvm

#endif
