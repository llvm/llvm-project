//===-- llvm/CodeGen/DwarfDebug.cpp - Dwarf Debug Framework ---------------===//
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

#define DEBUG_TYPE "dwarfdebug"
#include "DwarfDebug.h"
#include "DIE.h"
#include "DwarfAccelTable.h"
#include "DwarfCompileUnit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/DIBuilder.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

static cl::opt<bool> DisableDebugInfoPrinting("disable-debug-info-print",
                                              cl::Hidden,
     cl::desc("Disable debug info printing"));

static cl::opt<bool> UnknownLocations("use-unknown-locations", cl::Hidden,
     cl::desc("Make an absence of debug location information explicit."),
     cl::init(false));

static cl::opt<bool> GenerateDwarfPubNamesSection("generate-dwarf-pubnames",
     cl::Hidden, cl::init(false),
     cl::desc("Generate DWARF pubnames section"));

namespace {
  enum DefaultOnOff {
    Default, Enable, Disable
  };
}

static cl::opt<DefaultOnOff> DwarfAccelTables("dwarf-accel-tables", cl::Hidden,
     cl::desc("Output prototype dwarf accelerator tables."),
     cl::values(
                clEnumVal(Default, "Default for platform"),
                clEnumVal(Enable, "Enabled"),
                clEnumVal(Disable, "Disabled"),
                clEnumValEnd),
     cl::init(Default));

static cl::opt<DefaultOnOff> DarwinGDBCompat("darwin-gdb-compat", cl::Hidden,
     cl::desc("Compatibility with Darwin gdb."),
     cl::values(
                clEnumVal(Default, "Default for platform"),
                clEnumVal(Enable, "Enabled"),
                clEnumVal(Disable, "Disabled"),
                clEnumValEnd),
     cl::init(Default));

static cl::opt<DefaultOnOff> SplitDwarf("split-dwarf", cl::Hidden,
     cl::desc("Output prototype dwarf split debug info."),
     cl::values(
                clEnumVal(Default, "Default for platform"),
                clEnumVal(Enable, "Enabled"),
                clEnumVal(Disable, "Disabled"),
                clEnumValEnd),
     cl::init(Default));

namespace {
  const char *DWARFGroupName = "DWARF Emission";
  const char *DbgTimerName = "DWARF Debug Writer";

  struct CompareFirst {
    template <typename T> bool operator()(const T &lhs, const T &rhs) const {
      return lhs.first < rhs.first;
    }
  };
} // end anonymous namespace

//===----------------------------------------------------------------------===//

// Configuration values for initial hash set sizes (log2).
//
static const unsigned InitAbbreviationsSetSize = 9; // log2(512)

namespace llvm {

DIType DbgVariable::getType() const {
  DIType Ty = Var.getType();
  // FIXME: isBlockByrefVariable should be reformulated in terms of complex
  // addresses instead.
  if (Var.isBlockByrefVariable()) {
    /* Byref variables, in Blocks, are declared by the programmer as
       "SomeType VarName;", but the compiler creates a
       __Block_byref_x_VarName struct, and gives the variable VarName
       either the struct, or a pointer to the struct, as its type.  This
       is necessary for various behind-the-scenes things the compiler
       needs to do with by-reference variables in blocks.

       However, as far as the original *programmer* is concerned, the
       variable should still have type 'SomeType', as originally declared.

       The following function dives into the __Block_byref_x_VarName
       struct to find the original type of the variable.  This will be
       passed back to the code generating the type for the Debug
       Information Entry for the variable 'VarName'.  'VarName' will then
       have the original type 'SomeType' in its debug information.

       The original type 'SomeType' will be the type of the field named
       'VarName' inside the __Block_byref_x_VarName struct.

       NOTE: In order for this to not completely fail on the debugger
       side, the Debug Information Entry for the variable VarName needs to
       have a DW_AT_location that tells the debugger how to unwind through
       the pointers and __Block_byref_x_VarName struct to find the actual
       value of the variable.  The function addBlockByrefType does this.  */
    DIType subType = Ty;
    unsigned tag = Ty.getTag();

    if (tag == dwarf::DW_TAG_pointer_type) {
      DIDerivedType DTy = DIDerivedType(Ty);
      subType = DTy.getTypeDerivedFrom();
    }

    DICompositeType blockStruct = DICompositeType(subType);
    DIArray Elements = blockStruct.getTypeArray();

    for (unsigned i = 0, N = Elements.getNumElements(); i < N; ++i) {
      DIDescriptor Element = Elements.getElement(i);
      DIDerivedType DT = DIDerivedType(Element);
      if (getName() == DT.getName())
        return (DT.getTypeDerivedFrom());
    }
  }
  return Ty;
}

} // end llvm namespace

DwarfDebug::DwarfDebug(AsmPrinter *A, Module *M)
  : Asm(A), MMI(Asm->MMI), FirstCU(0),
    AbbreviationsSet(InitAbbreviationsSetSize),
    SourceIdMap(DIEValueAllocator),
    PrevLabel(NULL), GlobalCUIndexCount(0),
    InfoHolder(A, &AbbreviationsSet, &Abbreviations, "info_string",
               DIEValueAllocator),
    SkeletonAbbrevSet(InitAbbreviationsSetSize),
    SkeletonHolder(A, &SkeletonAbbrevSet, &SkeletonAbbrevs, "skel_string",
                   DIEValueAllocator) {

  DwarfInfoSectionSym = DwarfAbbrevSectionSym = 0;
  DwarfStrSectionSym = TextSectionSym = 0;
  DwarfDebugRangeSectionSym = DwarfDebugLocSectionSym = DwarfLineSectionSym = 0;
  DwarfAddrSectionSym = 0;
  DwarfAbbrevDWOSectionSym = DwarfStrDWOSectionSym = 0;
  FunctionBeginSym = FunctionEndSym = 0;

  // Turn on accelerator tables and older gdb compatibility
  // for Darwin.
  bool IsDarwin = Triple(A->getTargetTriple()).isOSDarwin();
  if (DarwinGDBCompat == Default) {
    if (IsDarwin)
      IsDarwinGDBCompat = true;
    else
      IsDarwinGDBCompat = false;
  } else
    IsDarwinGDBCompat = DarwinGDBCompat == Enable ? true : false;

  if (DwarfAccelTables == Default) {
    if (IsDarwin)
      HasDwarfAccelTables = true;
    else
      HasDwarfAccelTables = false;
  } else
    HasDwarfAccelTables = DwarfAccelTables == Enable ? true : false;

  if (SplitDwarf == Default)
    HasSplitDwarf = false;
  else
    HasSplitDwarf = SplitDwarf == Enable ? true : false;

  {
    NamedRegionTimer T(DbgTimerName, DWARFGroupName, TimePassesIsEnabled);
    beginModule();
  }
}
DwarfDebug::~DwarfDebug() {
}

// Switch to the specified MCSection and emit an assembler
// temporary label to it if SymbolStem is specified.
static MCSymbol *emitSectionSym(AsmPrinter *Asm, const MCSection *Section,
                                const char *SymbolStem = 0) {
  Asm->OutStreamer.SwitchSection(Section);
  if (!SymbolStem) return 0;

  MCSymbol *TmpSym = Asm->GetTempSymbol(SymbolStem);
  Asm->OutStreamer.EmitLabel(TmpSym);
  return TmpSym;
}

MCSymbol *DwarfUnits::getStringPoolSym() {
  return Asm->GetTempSymbol(StringPref);
}

MCSymbol *DwarfUnits::getStringPoolEntry(StringRef Str) {
  std::pair<MCSymbol*, unsigned> &Entry =
    StringPool.GetOrCreateValue(Str).getValue();
  if (Entry.first) return Entry.first;

  Entry.second = NextStringPoolNumber++;
  return Entry.first = Asm->GetTempSymbol(StringPref, Entry.second);
}

unsigned DwarfUnits::getStringPoolIndex(StringRef Str) {
  std::pair<MCSymbol*, unsigned> &Entry =
    StringPool.GetOrCreateValue(Str).getValue();
  if (Entry.first) return Entry.second;

  Entry.second = NextStringPoolNumber++;
  Entry.first = Asm->GetTempSymbol(StringPref, Entry.second);
  return Entry.second;
}

unsigned DwarfUnits::getAddrPoolIndex(MCSymbol *Sym) {
  std::pair<MCSymbol*, unsigned> &Entry = AddressPool[Sym];
  if (Entry.first) return Entry.second;

  Entry.second = NextAddrPoolNumber++;
  Entry.first = Sym;
  return Entry.second;
}

// Define a unique number for the abbreviation.
//
void DwarfUnits::assignAbbrevNumber(DIEAbbrev &Abbrev) {
  // Profile the node so that we can make it unique.
  FoldingSetNodeID ID;
  Abbrev.Profile(ID);

  // Check the set for priors.
  DIEAbbrev *InSet = AbbreviationsSet->GetOrInsertNode(&Abbrev);

  // If it's newly added.
  if (InSet == &Abbrev) {
    // Add to abbreviation list.
    Abbreviations->push_back(&Abbrev);

    // Assign the vector position + 1 as its number.
    Abbrev.setNumber(Abbreviations->size());
  } else {
    // Assign existing abbreviation number.
    Abbrev.setNumber(InSet->getNumber());
  }
}

// If special LLVM prefix that is used to inform the asm
// printer to not emit usual symbol prefix before the symbol name is used then
// return linkage name after skipping this special LLVM prefix.
static StringRef getRealLinkageName(StringRef LinkageName) {
  char One = '\1';
  if (LinkageName.startswith(StringRef(&One, 1)))
    return LinkageName.substr(1);
  return LinkageName;
}

static bool isObjCClass(StringRef Name) {
  return Name.startswith("+") || Name.startswith("-");
}

static bool hasObjCCategory(StringRef Name) {
  if (!isObjCClass(Name)) return false;

  size_t pos = Name.find(')');
  if (pos != std::string::npos) {
    if (Name[pos+1] != ' ') return false;
    return true;
  }
  return false;
}

static void getObjCClassCategory(StringRef In, StringRef &Class,
                                 StringRef &Category) {
  if (!hasObjCCategory(In)) {
    Class = In.slice(In.find('[') + 1, In.find(' '));
    Category = "";
    return;
  }

  Class = In.slice(In.find('[') + 1, In.find('('));
  Category = In.slice(In.find('[') + 1, In.find(' '));
  return;
}

static StringRef getObjCMethodName(StringRef In) {
  return In.slice(In.find(' ') + 1, In.find(']'));
}

// Add the various names to the Dwarf accelerator table names.
static void addSubprogramNames(CompileUnit *TheCU, DISubprogram SP,
                               DIE* Die) {
  if (!SP.isDefinition()) return;

  TheCU->addAccelName(SP.getName(), Die);

  // If the linkage name is different than the name, go ahead and output
  // that as well into the name table.
  if (SP.getLinkageName() != "" && SP.getName() != SP.getLinkageName())
    TheCU->addAccelName(SP.getLinkageName(), Die);

  // If this is an Objective-C selector name add it to the ObjC accelerator
  // too.
  if (isObjCClass(SP.getName())) {
    StringRef Class, Category;
    getObjCClassCategory(SP.getName(), Class, Category);
    TheCU->addAccelObjC(Class, Die);
    if (Category != "")
      TheCU->addAccelObjC(Category, Die);
    // Also add the base method name to the name table.
    TheCU->addAccelName(getObjCMethodName(SP.getName()), Die);
  }
}

// Find DIE for the given subprogram and attach appropriate DW_AT_low_pc
// and DW_AT_high_pc attributes. If there are global variables in this
// scope then create and insert DIEs for these variables.
DIE *DwarfDebug::updateSubprogramScopeDIE(CompileUnit *SPCU,
                                          const MDNode *SPNode) {
  DIE *SPDie = SPCU->getDIE(SPNode);

  assert(SPDie && "Unable to find subprogram DIE!");
  DISubprogram SP(SPNode);

  // If we're updating an abstract DIE, then we will be adding the children and
  // object pointer later on. But what we don't want to do is process the
  // concrete DIE twice.
  DIE *AbsSPDIE = AbstractSPDies.lookup(SPNode);
  if (AbsSPDIE) {
    bool InSameCU = (AbsSPDIE->getCompileUnit() == SPCU->getCUDie());
    // Pick up abstract subprogram DIE.
    SPDie = new DIE(dwarf::DW_TAG_subprogram);
    // If AbsSPDIE belongs to a different CU, use DW_FORM_ref_addr instead of
    // DW_FORM_ref4.
    SPCU->addDIEEntry(SPDie, dwarf::DW_AT_abstract_origin,
                      InSameCU ? dwarf::DW_FORM_ref4 : dwarf::DW_FORM_ref_addr,
                      AbsSPDIE);
    SPCU->addDie(SPDie);
  } else {
    DISubprogram SPDecl = SP.getFunctionDeclaration();
    if (!SPDecl.isSubprogram()) {
      // There is not any need to generate specification DIE for a function
      // defined at compile unit level. If a function is defined inside another
      // function then gdb prefers the definition at top level and but does not
      // expect specification DIE in parent function. So avoid creating
      // specification DIE for a function defined inside a function.
      if (SP.isDefinition() && !SP.getContext().isCompileUnit() &&
          !SP.getContext().isFile() &&
          !isSubprogramContext(SP.getContext())) {
        SPCU->addFlag(SPDie, dwarf::DW_AT_declaration);

        // Add arguments.
        DICompositeType SPTy = SP.getType();
        DIArray Args = SPTy.getTypeArray();
        unsigned SPTag = SPTy.getTag();
        if (SPTag == dwarf::DW_TAG_subroutine_type)
          for (unsigned i = 1, N = Args.getNumElements(); i < N; ++i) {
            DIE *Arg = new DIE(dwarf::DW_TAG_formal_parameter);
            DIType ATy = DIType(Args.getElement(i));
            SPCU->addType(Arg, ATy);
            if (ATy.isArtificial())
              SPCU->addFlag(Arg, dwarf::DW_AT_artificial);
            if (ATy.isObjectPointer())
              SPCU->addDIEEntry(SPDie, dwarf::DW_AT_object_pointer,
                                dwarf::DW_FORM_ref4, Arg);
            SPDie->addChild(Arg);
          }
        DIE *SPDeclDie = SPDie;
        SPDie = new DIE(dwarf::DW_TAG_subprogram);
        SPCU->addDIEEntry(SPDie, dwarf::DW_AT_specification,
                          dwarf::DW_FORM_ref4, SPDeclDie);
        SPCU->addDie(SPDie);
      }
    }
  }

  SPCU->addLabelAddress(SPDie, dwarf::DW_AT_low_pc,
                        Asm->GetTempSymbol("func_begin",
                                           Asm->getFunctionNumber()));
  SPCU->addLabelAddress(SPDie, dwarf::DW_AT_high_pc,
                        Asm->GetTempSymbol("func_end",
                                           Asm->getFunctionNumber()));
  const TargetRegisterInfo *RI = Asm->TM.getRegisterInfo();
  MachineLocation Location(RI->getFrameRegister(*Asm->MF));
  SPCU->addAddress(SPDie, dwarf::DW_AT_frame_base, Location);

  // Add name to the name table, we do this here because we're guaranteed
  // to have concrete versions of our DW_TAG_subprogram nodes.
  addSubprogramNames(SPCU, SP, SPDie);

  return SPDie;
}

// Construct new DW_TAG_lexical_block for this scope and attach
// DW_AT_low_pc/DW_AT_high_pc labels.
DIE *DwarfDebug::constructLexicalScopeDIE(CompileUnit *TheCU,
                                          LexicalScope *Scope) {
  DIE *ScopeDIE = new DIE(dwarf::DW_TAG_lexical_block);
  if (Scope->isAbstractScope())
    return ScopeDIE;

  const SmallVector<InsnRange, 4> &Ranges = Scope->getRanges();
  if (Ranges.empty())
    return 0;

  SmallVector<InsnRange, 4>::const_iterator RI = Ranges.begin();
  if (Ranges.size() > 1) {
    // .debug_range section has not been laid out yet. Emit offset in
    // .debug_range as a uint, size 4, for now. emitDIE will handle
    // DW_AT_ranges appropriately.
    TheCU->addUInt(ScopeDIE, dwarf::DW_AT_ranges, dwarf::DW_FORM_data4,
                   DebugRangeSymbols.size()
                   * Asm->getDataLayout().getPointerSize());
    for (SmallVector<InsnRange, 4>::const_iterator RI = Ranges.begin(),
         RE = Ranges.end(); RI != RE; ++RI) {
      DebugRangeSymbols.push_back(getLabelBeforeInsn(RI->first));
      DebugRangeSymbols.push_back(getLabelAfterInsn(RI->second));
    }
    DebugRangeSymbols.push_back(NULL);
    DebugRangeSymbols.push_back(NULL);
    return ScopeDIE;
  }

  MCSymbol *Start = getLabelBeforeInsn(RI->first);
  MCSymbol *End = getLabelAfterInsn(RI->second);

  if (End == 0) return 0;

  assert(Start->isDefined() && "Invalid starting label for an inlined scope!");
  assert(End->isDefined() && "Invalid end label for an inlined scope!");

  TheCU->addLabelAddress(ScopeDIE, dwarf::DW_AT_low_pc, Start);
  TheCU->addLabelAddress(ScopeDIE, dwarf::DW_AT_high_pc, End);

  return ScopeDIE;
}

// This scope represents inlined body of a function. Construct DIE to
// represent this concrete inlined copy of the function.
DIE *DwarfDebug::constructInlinedScopeDIE(CompileUnit *TheCU,
                                          LexicalScope *Scope) {
  const SmallVector<InsnRange, 4> &Ranges = Scope->getRanges();
  assert(Ranges.empty() == false &&
         "LexicalScope does not have instruction markers!");

  if (!Scope->getScopeNode())
    return NULL;
  DIScope DS(Scope->getScopeNode());
  DISubprogram InlinedSP = getDISubprogram(DS);
  DIE *OriginDIE = TheCU->getDIE(InlinedSP);
  if (!OriginDIE) {
    DEBUG(dbgs() << "Unable to find original DIE for an inlined subprogram.");
    return NULL;
  }

  SmallVector<InsnRange, 4>::const_iterator RI = Ranges.begin();
  MCSymbol *StartLabel = getLabelBeforeInsn(RI->first);
  MCSymbol *EndLabel = getLabelAfterInsn(RI->second);

  if (StartLabel == 0 || EndLabel == 0) {
    llvm_unreachable("Unexpected Start and End labels for an inlined scope!");
  }
  assert(StartLabel->isDefined() &&
         "Invalid starting label for an inlined scope!");
  assert(EndLabel->isDefined() &&
         "Invalid end label for an inlined scope!");

  DIE *ScopeDIE = new DIE(dwarf::DW_TAG_inlined_subroutine);
  TheCU->addDIEEntry(ScopeDIE, dwarf::DW_AT_abstract_origin,
                     dwarf::DW_FORM_ref4, OriginDIE);

  if (Ranges.size() > 1) {
    // .debug_range section has not been laid out yet. Emit offset in
    // .debug_range as a uint, size 4, for now. emitDIE will handle
    // DW_AT_ranges appropriately.
    TheCU->addUInt(ScopeDIE, dwarf::DW_AT_ranges, dwarf::DW_FORM_data4,
                   DebugRangeSymbols.size()
                   * Asm->getDataLayout().getPointerSize());
    for (SmallVector<InsnRange, 4>::const_iterator RI = Ranges.begin(),
         RE = Ranges.end(); RI != RE; ++RI) {
      DebugRangeSymbols.push_back(getLabelBeforeInsn(RI->first));
      DebugRangeSymbols.push_back(getLabelAfterInsn(RI->second));
    }
    DebugRangeSymbols.push_back(NULL);
    DebugRangeSymbols.push_back(NULL);
  } else {
    TheCU->addLabelAddress(ScopeDIE, dwarf::DW_AT_low_pc, StartLabel);
    TheCU->addLabelAddress(ScopeDIE, dwarf::DW_AT_high_pc, EndLabel);
  }

  InlinedSubprogramDIEs.insert(OriginDIE);

  // Track the start label for this inlined function.
  //.debug_inlined section specification does not clearly state how
  // to emit inlined scope that is split into multiple instruction ranges.
  // For now, use first instruction range and emit low_pc/high_pc pair and
  // corresponding .debug_inlined section entry for this pair.
  DenseMap<const MDNode *, SmallVector<InlineInfoLabels, 4> >::iterator
    I = InlineInfo.find(InlinedSP);

  if (I == InlineInfo.end()) {
    InlineInfo[InlinedSP].push_back(std::make_pair(StartLabel, ScopeDIE));
    InlinedSPNodes.push_back(InlinedSP);
  } else
    I->second.push_back(std::make_pair(StartLabel, ScopeDIE));

  DILocation DL(Scope->getInlinedAt());
  TheCU->addUInt(ScopeDIE, dwarf::DW_AT_call_file, 0,
                 getOrCreateSourceID(DL.getFilename(), DL.getDirectory(),
                                     TheCU->getUniqueID()));
  TheCU->addUInt(ScopeDIE, dwarf::DW_AT_call_line, 0, DL.getLineNumber());

  // Add name to the name table, we do this here because we're guaranteed
  // to have concrete versions of our DW_TAG_inlined_subprogram nodes.
  addSubprogramNames(TheCU, InlinedSP, ScopeDIE);

  return ScopeDIE;
}

// Construct a DIE for this scope.
DIE *DwarfDebug::constructScopeDIE(CompileUnit *TheCU, LexicalScope *Scope) {
  if (!Scope || !Scope->getScopeNode())
    return NULL;

  DIScope DS(Scope->getScopeNode());
  // Early return to avoid creating dangling variable|scope DIEs.
  if (!Scope->getInlinedAt() && DS.isSubprogram() && Scope->isAbstractScope() &&
      !TheCU->getDIE(DS))
    return NULL;

  SmallVector<DIE *, 8> Children;
  DIE *ObjectPointer = NULL;

  // Collect arguments for current function.
  if (LScopes.isCurrentFunctionScope(Scope))
    for (unsigned i = 0, N = CurrentFnArguments.size(); i < N; ++i)
      if (DbgVariable *ArgDV = CurrentFnArguments[i])
        if (DIE *Arg =
            TheCU->constructVariableDIE(ArgDV, Scope->isAbstractScope())) {
          Children.push_back(Arg);
          if (ArgDV->isObjectPointer()) ObjectPointer = Arg;
        }

  // Collect lexical scope children first.
  const SmallVector<DbgVariable *, 8> &Variables = ScopeVariables.lookup(Scope);
  for (unsigned i = 0, N = Variables.size(); i < N; ++i)
    if (DIE *Variable =
        TheCU->constructVariableDIE(Variables[i], Scope->isAbstractScope())) {
      Children.push_back(Variable);
      if (Variables[i]->isObjectPointer()) ObjectPointer = Variable;
    }
  const SmallVector<LexicalScope *, 4> &Scopes = Scope->getChildren();
  for (unsigned j = 0, M = Scopes.size(); j < M; ++j)
    if (DIE *Nested = constructScopeDIE(TheCU, Scopes[j]))
      Children.push_back(Nested);
  DIE *ScopeDIE = NULL;
  if (Scope->getInlinedAt())
    ScopeDIE = constructInlinedScopeDIE(TheCU, Scope);
  else if (DS.isSubprogram()) {
    ProcessedSPNodes.insert(DS);
    if (Scope->isAbstractScope()) {
      ScopeDIE = TheCU->getDIE(DS);
      // Note down abstract DIE.
      if (ScopeDIE)
        AbstractSPDies.insert(std::make_pair(DS, ScopeDIE));
    }
    else
      ScopeDIE = updateSubprogramScopeDIE(TheCU, DS);
  }
  else {
    // There is no need to emit empty lexical block DIE.
    std::pair<ImportedEntityMap::const_iterator,
              ImportedEntityMap::const_iterator> Range = std::equal_range(
        ScopesWithImportedEntities.begin(), ScopesWithImportedEntities.end(),
        std::pair<const MDNode *, const MDNode *>(DS, (const MDNode*)0),
        CompareFirst());
    if (Children.empty() && Range.first == Range.second)
      return NULL;
    ScopeDIE = constructLexicalScopeDIE(TheCU, Scope);
    for (ImportedEntityMap::const_iterator i = Range.first; i != Range.second; ++i)
      constructImportedModuleDIE(TheCU, i->second, ScopeDIE);
  }

  if (!ScopeDIE) return NULL;

  // Add children
  for (SmallVector<DIE *, 8>::iterator I = Children.begin(),
         E = Children.end(); I != E; ++I)
    ScopeDIE->addChild(*I);

  if (DS.isSubprogram() && ObjectPointer != NULL)
    TheCU->addDIEEntry(ScopeDIE, dwarf::DW_AT_object_pointer,
                       dwarf::DW_FORM_ref4, ObjectPointer);

  if (DS.isSubprogram())
    TheCU->addPubTypes(DISubprogram(DS));

  return ScopeDIE;
}

// Look up the source id with the given directory and source file names.
// If none currently exists, create a new id and insert it in the
// SourceIds map. This can update DirectoryNames and SourceFileNames maps
// as well.
unsigned DwarfDebug::getOrCreateSourceID(StringRef FileName,
                                         StringRef DirName, unsigned CUID) {
  // If we use .loc in assembly, we can't separate .file entries according to
  // compile units. Thus all files will belong to the default compile unit.
  if (Asm->TM.hasMCUseLoc() &&
      Asm->OutStreamer.getKind() == MCStreamer::SK_AsmStreamer)
    CUID = 0;

  // If FE did not provide a file name, then assume stdin.
  if (FileName.empty())
    return getOrCreateSourceID("<stdin>", StringRef(), CUID);

  // TODO: this might not belong here. See if we can factor this better.
  if (DirName == CompilationDir)
    DirName = "";

  // FileIDCUMap stores the current ID for the given compile unit.
  unsigned SrcId = FileIDCUMap[CUID] + 1;

  // We look up the CUID/file/dir by concatenating them with a zero byte.
  SmallString<128> NamePair;
  NamePair += utostr(CUID);
  NamePair += '\0';
  NamePair += DirName;
  NamePair += '\0'; // Zero bytes are not allowed in paths.
  NamePair += FileName;

  StringMapEntry<unsigned> &Ent = SourceIdMap.GetOrCreateValue(NamePair, SrcId);
  if (Ent.getValue() != SrcId)
    return Ent.getValue();

  FileIDCUMap[CUID] = SrcId;
  // Print out a .file directive to specify files for .loc directives.
  Asm->OutStreamer.EmitDwarfFileDirective(SrcId, DirName, FileName, CUID);

  return SrcId;
}

// Create new CompileUnit for the given metadata node with tag
// DW_TAG_compile_unit.
CompileUnit *DwarfDebug::constructCompileUnit(const MDNode *N) {
  DICompileUnit DIUnit(N);
  StringRef FN = DIUnit.getFilename();
  CompilationDir = DIUnit.getDirectory();

  DIE *Die = new DIE(dwarf::DW_TAG_compile_unit);
  CompileUnit *NewCU = new CompileUnit(GlobalCUIndexCount++,
                                       DIUnit.getLanguage(), Die, Asm,
                                       this, &InfoHolder);

  FileIDCUMap[NewCU->getUniqueID()] = 0;
  // Call this to emit a .file directive if it wasn't emitted for the source
  // file this CU comes from yet.
  getOrCreateSourceID(FN, CompilationDir, NewCU->getUniqueID());

  NewCU->addString(Die, dwarf::DW_AT_producer, DIUnit.getProducer());
  NewCU->addUInt(Die, dwarf::DW_AT_language, dwarf::DW_FORM_data2,
                 DIUnit.getLanguage());
  NewCU->addString(Die, dwarf::DW_AT_name, FN);

  // 2.17.1 requires that we use DW_AT_low_pc for a single entry point
  // into an entity. We're using 0 (or a NULL label) for this. For
  // split dwarf it's in the skeleton CU so omit it here.
  if (!useSplitDwarf())
    NewCU->addLabelAddress(Die, dwarf::DW_AT_low_pc, NULL);

  // Define start line table label for each Compile Unit.
  MCSymbol *LineTableStartSym = Asm->GetTempSymbol("line_table_start",
                                                   NewCU->getUniqueID());
  Asm->OutStreamer.getContext().setMCLineTableSymbol(LineTableStartSym,
                                                     NewCU->getUniqueID());

  // Use a single line table if we are using .loc and generating assembly.
  bool UseTheFirstCU =
    (Asm->TM.hasMCUseLoc() &&
     Asm->OutStreamer.getKind() == MCStreamer::SK_AsmStreamer) ||
    (NewCU->getUniqueID() == 0);

  // DW_AT_stmt_list is a offset of line number information for this
  // compile unit in debug_line section. For split dwarf this is
  // left in the skeleton CU and so not included.
  // The line table entries are not always emitted in assembly, so it
  // is not okay to use line_table_start here.
  if (!useSplitDwarf()) {
    if (Asm->MAI->doesDwarfUseRelocationsAcrossSections())
      NewCU->addLabel(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_data4,
                      UseTheFirstCU ?
                      Asm->GetTempSymbol("section_line") : LineTableStartSym);
    else if (UseTheFirstCU)
      NewCU->addUInt(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_data4, 0);
    else
      NewCU->addDelta(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_data4,
                      LineTableStartSym, DwarfLineSectionSym);
  }

  // If we're using split dwarf the compilation dir is going to be in the
  // skeleton CU and so we don't need to duplicate it here.
  if (!useSplitDwarf() && !CompilationDir.empty())
    NewCU->addString(Die, dwarf::DW_AT_comp_dir, CompilationDir);
  if (DIUnit.isOptimized())
    NewCU->addFlag(Die, dwarf::DW_AT_APPLE_optimized);

  StringRef Flags = DIUnit.getFlags();
  if (!Flags.empty())
    NewCU->addString(Die, dwarf::DW_AT_APPLE_flags, Flags);

  if (unsigned RVer = DIUnit.getRunTimeVersion())
    NewCU->addUInt(Die, dwarf::DW_AT_APPLE_major_runtime_vers,
            dwarf::DW_FORM_data1, RVer);

  if (!FirstCU)
    FirstCU = NewCU;

  InfoHolder.addUnit(NewCU);

  CUMap.insert(std::make_pair(N, NewCU));
  return NewCU;
}

// Construct subprogram DIE.
void DwarfDebug::constructSubprogramDIE(CompileUnit *TheCU,
                                        const MDNode *N) {
  CompileUnit *&CURef = SPMap[N];
  if (CURef)
    return;
  CURef = TheCU;

  DISubprogram SP(N);
  if (!SP.isDefinition())
    // This is a method declaration which will be handled while constructing
    // class type.
    return;

  DIE *SubprogramDie = TheCU->getOrCreateSubprogramDIE(SP);

  // Add to map.
  TheCU->insertDIE(N, SubprogramDie);

  // Add to context owner.
  TheCU->addToContextOwner(SubprogramDie, SP.getContext());

  // Expose as global, if requested.
  if (GenerateDwarfPubNamesSection)
    TheCU->addGlobalName(SP.getName(), SubprogramDie);
}

void DwarfDebug::constructImportedModuleDIE(CompileUnit *TheCU,
                                            const MDNode *N) {
  DIImportedModule Module(N);
  if (!Module.Verify())
    return;
  if (DIE *D = TheCU->getOrCreateContextDIE(Module.getContext()))
    constructImportedModuleDIE(TheCU, Module, D);
}

void DwarfDebug::constructImportedModuleDIE(CompileUnit *TheCU, const MDNode *N,
                                            DIE *Context) {
  DIImportedModule Module(N);
  if (!Module.Verify())
    return;
  return constructImportedModuleDIE(TheCU, Module, Context);
}

void DwarfDebug::constructImportedModuleDIE(CompileUnit *TheCU,
                                            const DIImportedModule &Module,
                                            DIE *Context) {
  assert(Module.Verify() &&
         "Use one of the MDNode * overloads to handle invalid metadata");
  assert(Context && "Should always have a context for an imported_module");
  DIE *IMDie = new DIE(dwarf::DW_TAG_imported_module);
  TheCU->insertDIE(Module, IMDie);
  DIE *NSDie = TheCU->getOrCreateNameSpace(Module.getNameSpace());
  unsigned FileID = getOrCreateSourceID(Module.getContext().getFilename(),
                                        Module.getContext().getDirectory(),
                                        TheCU->getUniqueID());
  TheCU->addUInt(IMDie, dwarf::DW_AT_decl_file, 0, FileID);
  TheCU->addUInt(IMDie, dwarf::DW_AT_decl_line, 0, Module.getLineNumber());
  TheCU->addDIEEntry(IMDie, dwarf::DW_AT_import, dwarf::DW_FORM_ref4, NSDie);
  Context->addChild(IMDie);
}

// Emit all Dwarf sections that should come prior to the content. Create
// global DIEs and emit initial debug info sections. This is invoked by
// the target AsmPrinter.
void DwarfDebug::beginModule() {
  if (DisableDebugInfoPrinting)
    return;

  const Module *M = MMI->getModule();

  // If module has named metadata anchors then use them, otherwise scan the
  // module using debug info finder to collect debug info.
  NamedMDNode *CU_Nodes = M->getNamedMetadata("llvm.dbg.cu");
  if (!CU_Nodes)
    return;

  // Emit initial sections so we can reference labels later.
  emitSectionLabels();

  for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
    DICompileUnit CUNode(CU_Nodes->getOperand(i));
    CompileUnit *CU = constructCompileUnit(CUNode);
    DIArray ImportedModules = CUNode.getImportedModules();
    for (unsigned i = 0, e = ImportedModules.getNumElements(); i != e; ++i)
      ScopesWithImportedEntities.push_back(std::make_pair(
          DIImportedModule(ImportedModules.getElement(i)).getContext(),
          ImportedModules.getElement(i)));
    std::sort(ScopesWithImportedEntities.begin(),
              ScopesWithImportedEntities.end(), CompareFirst());
    DIArray GVs = CUNode.getGlobalVariables();
    for (unsigned i = 0, e = GVs.getNumElements(); i != e; ++i)
      CU->createGlobalVariableDIE(GVs.getElement(i));
    DIArray SPs = CUNode.getSubprograms();
    for (unsigned i = 0, e = SPs.getNumElements(); i != e; ++i)
      constructSubprogramDIE(CU, SPs.getElement(i));
    DIArray EnumTypes = CUNode.getEnumTypes();
    for (unsigned i = 0, e = EnumTypes.getNumElements(); i != e; ++i)
      CU->getOrCreateTypeDIE(EnumTypes.getElement(i));
    DIArray RetainedTypes = CUNode.getRetainedTypes();
    for (unsigned i = 0, e = RetainedTypes.getNumElements(); i != e; ++i)
      CU->getOrCreateTypeDIE(RetainedTypes.getElement(i));
    // Emit imported_modules last so that the relevant context is already
    // available.
    for (unsigned i = 0, e = ImportedModules.getNumElements(); i != e; ++i)
      constructImportedModuleDIE(CU, ImportedModules.getElement(i));
    // If we're splitting the dwarf out now that we've got the entire
    // CU then construct a skeleton CU based upon it.
    if (useSplitDwarf()) {
      // This should be a unique identifier when we want to build .dwp files.
      CU->addUInt(CU->getCUDie(), dwarf::DW_AT_GNU_dwo_id,
                  dwarf::DW_FORM_data8, 0);
      // Now construct the skeleton CU associated.
      constructSkeletonCU(CUNode);
    }
  }

  // Tell MMI that we have debug info.
  MMI->setDebugInfoAvailability(true);

  // Prime section data.
  SectionMap.insert(Asm->getObjFileLowering().getTextSection());
}

// Attach DW_AT_inline attribute with inlined subprogram DIEs.
void DwarfDebug::computeInlinedDIEs() {
  // Attach DW_AT_inline attribute with inlined subprogram DIEs.
  for (SmallPtrSet<DIE *, 4>::iterator AI = InlinedSubprogramDIEs.begin(),
         AE = InlinedSubprogramDIEs.end(); AI != AE; ++AI) {
    DIE *ISP = *AI;
    FirstCU->addUInt(ISP, dwarf::DW_AT_inline, 0, dwarf::DW_INL_inlined);
  }
  for (DenseMap<const MDNode *, DIE *>::iterator AI = AbstractSPDies.begin(),
         AE = AbstractSPDies.end(); AI != AE; ++AI) {
    DIE *ISP = AI->second;
    if (InlinedSubprogramDIEs.count(ISP))
      continue;
    FirstCU->addUInt(ISP, dwarf::DW_AT_inline, 0, dwarf::DW_INL_inlined);
  }
}

// Collect info for variables that were optimized out.
void DwarfDebug::collectDeadVariables() {
  const Module *M = MMI->getModule();
  DenseMap<const MDNode *, LexicalScope *> DeadFnScopeMap;

  if (NamedMDNode *CU_Nodes = M->getNamedMetadata("llvm.dbg.cu")) {
    for (unsigned i = 0, e = CU_Nodes->getNumOperands(); i != e; ++i) {
      DICompileUnit TheCU(CU_Nodes->getOperand(i));
      DIArray Subprograms = TheCU.getSubprograms();
      for (unsigned i = 0, e = Subprograms.getNumElements(); i != e; ++i) {
        DISubprogram SP(Subprograms.getElement(i));
        if (ProcessedSPNodes.count(SP) != 0) continue;
        if (!SP.Verify()) continue;
        if (!SP.isDefinition()) continue;
        DIArray Variables = SP.getVariables();
        if (Variables.getNumElements() == 0) continue;

        LexicalScope *Scope =
          new LexicalScope(NULL, DIDescriptor(SP), NULL, false);
        DeadFnScopeMap[SP] = Scope;

        // Construct subprogram DIE and add variables DIEs.
        CompileUnit *SPCU = CUMap.lookup(TheCU);
        assert(SPCU && "Unable to find Compile Unit!");
        constructSubprogramDIE(SPCU, SP);
        DIE *ScopeDIE = SPCU->getDIE(SP);
        for (unsigned vi = 0, ve = Variables.getNumElements(); vi != ve; ++vi) {
          DIVariable DV(Variables.getElement(vi));
          if (!DV.Verify()) continue;
          DbgVariable *NewVar = new DbgVariable(DV, NULL);
          if (DIE *VariableDIE =
              SPCU->constructVariableDIE(NewVar, Scope->isAbstractScope()))
            ScopeDIE->addChild(VariableDIE);
        }
      }
    }
  }
  DeleteContainerSeconds(DeadFnScopeMap);
}

void DwarfDebug::finalizeModuleInfo() {
  // Collect info for variables that were optimized out.
  collectDeadVariables();

  // Attach DW_AT_inline attribute with inlined subprogram DIEs.
  computeInlinedDIEs();

  // Emit DW_AT_containing_type attribute to connect types with their
  // vtable holding type.
  for (DenseMap<const MDNode *, CompileUnit *>::iterator CUI = CUMap.begin(),
         CUE = CUMap.end(); CUI != CUE; ++CUI) {
    CompileUnit *TheCU = CUI->second;
    TheCU->constructContainingTypeDIEs();
  }

   // Compute DIE offsets and sizes.
  InfoHolder.computeSizeAndOffsets();
  if (useSplitDwarf())
    SkeletonHolder.computeSizeAndOffsets();
}

void DwarfDebug::endSections() {
  // Standard sections final addresses.
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getTextSection());
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("text_end"));
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering().getDataSection());
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("data_end"));

  // End text sections.
  for (unsigned I = 0, E = SectionMap.size(); I != E; ++I) {
    Asm->OutStreamer.SwitchSection(SectionMap[I]);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("section_end", I+1));
  }
}

// Emit all Dwarf sections that should come after the content.
void DwarfDebug::endModule() {

  if (!FirstCU) return;

  // End any existing sections.
  // TODO: Does this need to happen?
  endSections();

  // Finalize the debug info for the module.
  finalizeModuleInfo();

  if (!useSplitDwarf()) {
    // Emit all the DIEs into a debug info section.
    emitDebugInfo();

    // Corresponding abbreviations into a abbrev section.
    emitAbbreviations();

    // Emit info into a debug loc section.
    emitDebugLoc();

    // Emit info into a debug aranges section.
    emitDebugARanges();

    // Emit info into a debug ranges section.
    emitDebugRanges();

    // Emit info into a debug macinfo section.
    emitDebugMacInfo();

    // Emit inline info.
    // TODO: When we don't need the option anymore we
    // can remove all of the code that this section
    // depends upon.
    if (useDarwinGDBCompat())
      emitDebugInlineInfo();
  } else {
    // TODO: Fill this in for separated debug sections and separate
    // out information into new sections.

    // Emit the debug info section and compile units.
    emitDebugInfo();
    emitDebugInfoDWO();

    // Corresponding abbreviations into a abbrev section.
    emitAbbreviations();
    emitDebugAbbrevDWO();

    // Emit info into a debug loc section.
    emitDebugLoc();

    // Emit info into a debug aranges section.
    emitDebugARanges();

    // Emit info into a debug ranges section.
    emitDebugRanges();

    // Emit info into a debug macinfo section.
    emitDebugMacInfo();

    // Emit DWO addresses.
    InfoHolder.emitAddresses(Asm->getObjFileLowering().getDwarfAddrSection());

    // Emit inline info.
    // TODO: When we don't need the option anymore we
    // can remove all of the code that this section
    // depends upon.
    if (useDarwinGDBCompat())
      emitDebugInlineInfo();
  }

  // Emit info into the dwarf accelerator table sections.
  if (useDwarfAccelTables()) {
    emitAccelNames();
    emitAccelObjC();
    emitAccelNamespaces();
    emitAccelTypes();
  }

  // Emit info into a debug pubnames section, if requested.
  if (GenerateDwarfPubNamesSection)
    emitDebugPubnames();

  // Emit info into a debug pubtypes section.
  // TODO: When we don't need the option anymore we can
  // remove all of the code that adds to the table.
  if (useDarwinGDBCompat())
    emitDebugPubTypes();

  // Finally emit string information into a string table.
  emitDebugStr();
  if (useSplitDwarf())
    emitDebugStrDWO();

  // clean up.
  SPMap.clear();
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I)
    delete I->second;

  for (SmallVector<CompileUnit *, 1>::iterator I = SkeletonCUs.begin(),
         E = SkeletonCUs.end(); I != E; ++I)
    delete *I;

  // Reset these for the next Module if we have one.
  FirstCU = NULL;
}

// Find abstract variable, if any, associated with Var.
DbgVariable *DwarfDebug::findAbstractVariable(DIVariable &DV,
                                              DebugLoc ScopeLoc) {
  LLVMContext &Ctx = DV->getContext();
  // More then one inlined variable corresponds to one abstract variable.
  DIVariable Var = cleanseInlinedVariable(DV, Ctx);
  DbgVariable *AbsDbgVariable = AbstractVariables.lookup(Var);
  if (AbsDbgVariable)
    return AbsDbgVariable;

  LexicalScope *Scope = LScopes.findAbstractScope(ScopeLoc.getScope(Ctx));
  if (!Scope)
    return NULL;

  AbsDbgVariable = new DbgVariable(Var, NULL);
  addScopeVariable(Scope, AbsDbgVariable);
  AbstractVariables[Var] = AbsDbgVariable;
  return AbsDbgVariable;
}

// If Var is a current function argument then add it to CurrentFnArguments list.
bool DwarfDebug::addCurrentFnArgument(const MachineFunction *MF,
                                      DbgVariable *Var, LexicalScope *Scope) {
  if (!LScopes.isCurrentFunctionScope(Scope))
    return false;
  DIVariable DV = Var->getVariable();
  if (DV.getTag() != dwarf::DW_TAG_arg_variable)
    return false;
  unsigned ArgNo = DV.getArgNumber();
  if (ArgNo == 0)
    return false;

  size_t Size = CurrentFnArguments.size();
  if (Size == 0)
    CurrentFnArguments.resize(MF->getFunction()->arg_size());
  // llvm::Function argument size is not good indicator of how many
  // arguments does the function have at source level.
  if (ArgNo > Size)
    CurrentFnArguments.resize(ArgNo * 2);
  CurrentFnArguments[ArgNo - 1] = Var;
  return true;
}

// Collect variable information from side table maintained by MMI.
void
DwarfDebug::collectVariableInfoFromMMITable(const MachineFunction *MF,
                                   SmallPtrSet<const MDNode *, 16> &Processed) {
  MachineModuleInfo::VariableDbgInfoMapTy &VMap = MMI->getVariableDbgInfo();
  for (MachineModuleInfo::VariableDbgInfoMapTy::iterator VI = VMap.begin(),
         VE = VMap.end(); VI != VE; ++VI) {
    const MDNode *Var = VI->first;
    if (!Var) continue;
    Processed.insert(Var);
    DIVariable DV(Var);
    const std::pair<unsigned, DebugLoc> &VP = VI->second;

    LexicalScope *Scope = LScopes.findLexicalScope(VP.second);

    // If variable scope is not found then skip this variable.
    if (Scope == 0)
      continue;

    DbgVariable *AbsDbgVariable = findAbstractVariable(DV, VP.second);
    DbgVariable *RegVar = new DbgVariable(DV, AbsDbgVariable);
    RegVar->setFrameIndex(VP.first);
    if (!addCurrentFnArgument(MF, RegVar, Scope))
      addScopeVariable(Scope, RegVar);
    if (AbsDbgVariable)
      AbsDbgVariable->setFrameIndex(VP.first);
  }
}

// Return true if debug value, encoded by DBG_VALUE instruction, is in a
// defined reg.
static bool isDbgValueInDefinedReg(const MachineInstr *MI) {
  assert(MI->isDebugValue() && "Invalid DBG_VALUE machine instruction!");
  return MI->getNumOperands() == 3 &&
         MI->getOperand(0).isReg() && MI->getOperand(0).getReg() &&
         MI->getOperand(1).isImm() && MI->getOperand(1).getImm() == 0;
}

// Get .debug_loc entry for the instruction range starting at MI.
static DotDebugLocEntry getDebugLocEntry(AsmPrinter *Asm,
                                         const MCSymbol *FLabel,
                                         const MCSymbol *SLabel,
                                         const MachineInstr *MI) {
  const MDNode *Var =  MI->getOperand(MI->getNumOperands() - 1).getMetadata();

  if (MI->getNumOperands() != 3) {
    MachineLocation MLoc = Asm->getDebugValueLocation(MI);
    return DotDebugLocEntry(FLabel, SLabel, MLoc, Var);
  }
  if (MI->getOperand(0).isReg() && MI->getOperand(1).isImm()) {
    MachineLocation MLoc;
    // TODO: Currently an offset of 0 in a DBG_VALUE means
    // we need to generate a direct register value.
    // There is no way to specify an indirect value with offset 0.
    if (MI->getOperand(1).getImm() == 0)
      MLoc.set(MI->getOperand(0).getReg());
    else
      MLoc.set(MI->getOperand(0).getReg(), MI->getOperand(1).getImm());
    return DotDebugLocEntry(FLabel, SLabel, MLoc, Var);
  }
  if (MI->getOperand(0).isImm())
    return DotDebugLocEntry(FLabel, SLabel, MI->getOperand(0).getImm());
  if (MI->getOperand(0).isFPImm())
    return DotDebugLocEntry(FLabel, SLabel, MI->getOperand(0).getFPImm());
  if (MI->getOperand(0).isCImm())
    return DotDebugLocEntry(FLabel, SLabel, MI->getOperand(0).getCImm());

  llvm_unreachable("Unexpected 3 operand DBG_VALUE instruction!");
}

// Find variables for each lexical scope.
void
DwarfDebug::collectVariableInfo(const MachineFunction *MF,
                                SmallPtrSet<const MDNode *, 16> &Processed) {

  // collection info from MMI table.
  collectVariableInfoFromMMITable(MF, Processed);

  for (SmallVectorImpl<const MDNode*>::const_iterator
         UVI = UserVariables.begin(), UVE = UserVariables.end(); UVI != UVE;
         ++UVI) {
    const MDNode *Var = *UVI;
    if (Processed.count(Var))
      continue;

    // History contains relevant DBG_VALUE instructions for Var and instructions
    // clobbering it.
    SmallVectorImpl<const MachineInstr*> &History = DbgValues[Var];
    if (History.empty())
      continue;
    const MachineInstr *MInsn = History.front();

    DIVariable DV(Var);
    LexicalScope *Scope = NULL;
    if (DV.getTag() == dwarf::DW_TAG_arg_variable &&
        DISubprogram(DV.getContext()).describes(MF->getFunction()))
      Scope = LScopes.getCurrentFunctionScope();
    else if (MDNode *IA = DV.getInlinedAt())
      Scope = LScopes.findInlinedScope(DebugLoc::getFromDILocation(IA));
    else
      Scope = LScopes.findLexicalScope(cast<MDNode>(DV->getOperand(1)));
    // If variable scope is not found then skip this variable.
    if (!Scope)
      continue;

    Processed.insert(DV);
    assert(MInsn->isDebugValue() && "History must begin with debug value");
    DbgVariable *AbsVar = findAbstractVariable(DV, MInsn->getDebugLoc());
    DbgVariable *RegVar = new DbgVariable(DV, AbsVar);
    if (!addCurrentFnArgument(MF, RegVar, Scope))
      addScopeVariable(Scope, RegVar);
    if (AbsVar)
      AbsVar->setMInsn(MInsn);

    // Simplify ranges that are fully coalesced.
    if (History.size() <= 1 || (History.size() == 2 &&
                                MInsn->isIdenticalTo(History.back()))) {
      RegVar->setMInsn(MInsn);
      continue;
    }

    // Handle multiple DBG_VALUE instructions describing one variable.
    RegVar->setDotDebugLocOffset(DotDebugLocEntries.size());

    for (SmallVectorImpl<const MachineInstr*>::const_iterator
           HI = History.begin(), HE = History.end(); HI != HE; ++HI) {
      const MachineInstr *Begin = *HI;
      assert(Begin->isDebugValue() && "Invalid History entry");

      // Check if DBG_VALUE is truncating a range.
      if (Begin->getNumOperands() > 1 && Begin->getOperand(0).isReg()
          && !Begin->getOperand(0).getReg())
        continue;

      // Compute the range for a register location.
      const MCSymbol *FLabel = getLabelBeforeInsn(Begin);
      const MCSymbol *SLabel = 0;

      if (HI + 1 == HE)
        // If Begin is the last instruction in History then its value is valid
        // until the end of the function.
        SLabel = FunctionEndSym;
      else {
        const MachineInstr *End = HI[1];
        DEBUG(dbgs() << "DotDebugLoc Pair:\n"
              << "\t" << *Begin << "\t" << *End << "\n");
        if (End->isDebugValue())
          SLabel = getLabelBeforeInsn(End);
        else {
          // End is a normal instruction clobbering the range.
          SLabel = getLabelAfterInsn(End);
          assert(SLabel && "Forgot label after clobber instruction");
          ++HI;
        }
      }

      // The value is valid until the next DBG_VALUE or clobber.
      DotDebugLocEntries.push_back(getDebugLocEntry(Asm, FLabel, SLabel,
                                                    Begin));
    }
    DotDebugLocEntries.push_back(DotDebugLocEntry());
  }

  // Collect info for variables that were optimized out.
  LexicalScope *FnScope = LScopes.getCurrentFunctionScope();
  DIArray Variables = DISubprogram(FnScope->getScopeNode()).getVariables();
  for (unsigned i = 0, e = Variables.getNumElements(); i != e; ++i) {
    DIVariable DV(Variables.getElement(i));
    if (!DV || !DV.Verify() || !Processed.insert(DV))
      continue;
    if (LexicalScope *Scope = LScopes.findLexicalScope(DV.getContext()))
      addScopeVariable(Scope, new DbgVariable(DV, NULL));
  }
}

// Return Label preceding the instruction.
MCSymbol *DwarfDebug::getLabelBeforeInsn(const MachineInstr *MI) {
  MCSymbol *Label = LabelsBeforeInsn.lookup(MI);
  assert(Label && "Didn't insert label before instruction");
  return Label;
}

// Return Label immediately following the instruction.
MCSymbol *DwarfDebug::getLabelAfterInsn(const MachineInstr *MI) {
  return LabelsAfterInsn.lookup(MI);
}

// Process beginning of an instruction.
void DwarfDebug::beginInstruction(const MachineInstr *MI) {
  // Check if source location changes, but ignore DBG_VALUE locations.
  if (!MI->isDebugValue()) {
    DebugLoc DL = MI->getDebugLoc();
    if (DL != PrevInstLoc && (!DL.isUnknown() || UnknownLocations)) {
      unsigned Flags = 0;
      PrevInstLoc = DL;
      if (DL == PrologEndLoc) {
        Flags |= DWARF2_FLAG_PROLOGUE_END;
        PrologEndLoc = DebugLoc();
      }
      if (PrologEndLoc.isUnknown())
        Flags |= DWARF2_FLAG_IS_STMT;

      if (!DL.isUnknown()) {
        const MDNode *Scope = DL.getScope(Asm->MF->getFunction()->getContext());
        recordSourceLine(DL.getLine(), DL.getCol(), Scope, Flags);
      } else
        recordSourceLine(0, 0, 0, 0);
    }
  }

  // Insert labels where requested.
  DenseMap<const MachineInstr*, MCSymbol*>::iterator I =
    LabelsBeforeInsn.find(MI);

  // No label needed.
  if (I == LabelsBeforeInsn.end())
    return;

  // Label already assigned.
  if (I->second)
    return;

  if (!PrevLabel) {
    PrevLabel = MMI->getContext().CreateTempSymbol();
    Asm->OutStreamer.EmitLabel(PrevLabel);
  }
  I->second = PrevLabel;
}

// Process end of an instruction.
void DwarfDebug::endInstruction(const MachineInstr *MI) {
  // Don't create a new label after DBG_VALUE instructions.
  // They don't generate code.
  if (!MI->isDebugValue())
    PrevLabel = 0;

  DenseMap<const MachineInstr*, MCSymbol*>::iterator I =
    LabelsAfterInsn.find(MI);

  // No label needed.
  if (I == LabelsAfterInsn.end())
    return;

  // Label already assigned.
  if (I->second)
    return;

  // We need a label after this instruction.
  if (!PrevLabel) {
    PrevLabel = MMI->getContext().CreateTempSymbol();
    Asm->OutStreamer.EmitLabel(PrevLabel);
  }
  I->second = PrevLabel;
}

// Each LexicalScope has first instruction and last instruction to mark
// beginning and end of a scope respectively. Create an inverse map that list
// scopes starts (and ends) with an instruction. One instruction may start (or
// end) multiple scopes. Ignore scopes that are not reachable.
void DwarfDebug::identifyScopeMarkers() {
  SmallVector<LexicalScope *, 4> WorkList;
  WorkList.push_back(LScopes.getCurrentFunctionScope());
  while (!WorkList.empty()) {
    LexicalScope *S = WorkList.pop_back_val();

    const SmallVector<LexicalScope *, 4> &Children = S->getChildren();
    if (!Children.empty())
      for (SmallVector<LexicalScope *, 4>::const_iterator SI = Children.begin(),
             SE = Children.end(); SI != SE; ++SI)
        WorkList.push_back(*SI);

    if (S->isAbstractScope())
      continue;

    const SmallVector<InsnRange, 4> &Ranges = S->getRanges();
    if (Ranges.empty())
      continue;
    for (SmallVector<InsnRange, 4>::const_iterator RI = Ranges.begin(),
           RE = Ranges.end(); RI != RE; ++RI) {
      assert(RI->first && "InsnRange does not have first instruction!");
      assert(RI->second && "InsnRange does not have second instruction!");
      requestLabelBeforeInsn(RI->first);
      requestLabelAfterInsn(RI->second);
    }
  }
}

// Get MDNode for DebugLoc's scope.
static MDNode *getScopeNode(DebugLoc DL, const LLVMContext &Ctx) {
  if (MDNode *InlinedAt = DL.getInlinedAt(Ctx))
    return getScopeNode(DebugLoc::getFromDILocation(InlinedAt), Ctx);
  return DL.getScope(Ctx);
}

// Walk up the scope chain of given debug loc and find line number info
// for the function.
static DebugLoc getFnDebugLoc(DebugLoc DL, const LLVMContext &Ctx) {
  const MDNode *Scope = getScopeNode(DL, Ctx);
  DISubprogram SP = getDISubprogram(Scope);
  if (SP.Verify()) {
    // Check for number of operands since the compatibility is
    // cheap here.
    if (SP->getNumOperands() > 19)
      return DebugLoc::get(SP.getScopeLineNumber(), 0, SP);
    else
      return DebugLoc::get(SP.getLineNumber(), 0, SP);
  }

  return DebugLoc();
}

// Gather pre-function debug information.  Assumes being called immediately
// after the function entry point has been emitted.
void DwarfDebug::beginFunction(const MachineFunction *MF) {
  if (!MMI->hasDebugInfo()) return;
  LScopes.initialize(*MF);
  if (LScopes.empty()) return;
  identifyScopeMarkers();

  // Set DwarfCompileUnitID in MCContext to the Compile Unit this function
  // belongs to.
  LexicalScope *FnScope = LScopes.getCurrentFunctionScope();
  CompileUnit *TheCU = SPMap.lookup(FnScope->getScopeNode());
  assert(TheCU && "Unable to find compile unit!");
  if (Asm->TM.hasMCUseLoc() &&
      Asm->OutStreamer.getKind() == MCStreamer::SK_AsmStreamer)
    // Use a single line table if we are using .loc and generating assembly.
    Asm->OutStreamer.getContext().setDwarfCompileUnitID(0);
  else
    Asm->OutStreamer.getContext().setDwarfCompileUnitID(TheCU->getUniqueID());

  FunctionBeginSym = Asm->GetTempSymbol("func_begin",
                                        Asm->getFunctionNumber());
  // Assumes in correct section after the entry point.
  Asm->OutStreamer.EmitLabel(FunctionBeginSym);

  assert(UserVariables.empty() && DbgValues.empty() && "Maps weren't cleaned");

  const TargetRegisterInfo *TRI = Asm->TM.getRegisterInfo();
  // LiveUserVar - Map physreg numbers to the MDNode they contain.
  std::vector<const MDNode*> LiveUserVar(TRI->getNumRegs());

  for (MachineFunction::const_iterator I = MF->begin(), E = MF->end();
       I != E; ++I) {
    bool AtBlockEntry = true;
    for (MachineBasicBlock::const_iterator II = I->begin(), IE = I->end();
         II != IE; ++II) {
      const MachineInstr *MI = II;

      if (MI->isDebugValue()) {
        assert(MI->getNumOperands() > 1 && "Invalid machine instruction!");

        // Keep track of user variables.
        const MDNode *Var =
          MI->getOperand(MI->getNumOperands() - 1).getMetadata();

        // Variable is in a register, we need to check for clobbers.
        if (isDbgValueInDefinedReg(MI))
          LiveUserVar[MI->getOperand(0).getReg()] = Var;

        // Check the history of this variable.
        SmallVectorImpl<const MachineInstr*> &History = DbgValues[Var];
        if (History.empty()) {
          UserVariables.push_back(Var);
          // The first mention of a function argument gets the FunctionBeginSym
          // label, so arguments are visible when breaking at function entry.
          DIVariable DV(Var);
          if (DV.Verify() && DV.getTag() == dwarf::DW_TAG_arg_variable &&
              DISubprogram(getDISubprogram(DV.getContext()))
                .describes(MF->getFunction()))
            LabelsBeforeInsn[MI] = FunctionBeginSym;
        } else {
          // We have seen this variable before. Try to coalesce DBG_VALUEs.
          const MachineInstr *Prev = History.back();
          if (Prev->isDebugValue()) {
            // Coalesce identical entries at the end of History.
            if (History.size() >= 2 &&
                Prev->isIdenticalTo(History[History.size() - 2])) {
              DEBUG(dbgs() << "Coalescing identical DBG_VALUE entries:\n"
                    << "\t" << *Prev
                    << "\t" << *History[History.size() - 2] << "\n");
              History.pop_back();
            }

            // Terminate old register assignments that don't reach MI;
            MachineFunction::const_iterator PrevMBB = Prev->getParent();
            if (PrevMBB != I && (!AtBlockEntry || llvm::next(PrevMBB) != I) &&
                isDbgValueInDefinedReg(Prev)) {
              // Previous register assignment needs to terminate at the end of
              // its basic block.
              MachineBasicBlock::const_iterator LastMI =
                PrevMBB->getLastNonDebugInstr();
              if (LastMI == PrevMBB->end()) {
                // Drop DBG_VALUE for empty range.
                DEBUG(dbgs() << "Dropping DBG_VALUE for empty range:\n"
                      << "\t" << *Prev << "\n");
                History.pop_back();
              }
              else {
                // Terminate after LastMI.
                History.push_back(LastMI);
              }
            }
          }
        }
        History.push_back(MI);
      } else {
        // Not a DBG_VALUE instruction.
        if (!MI->isLabel())
          AtBlockEntry = false;

        // First known non-DBG_VALUE and non-frame setup location marks
        // the beginning of the function body.
        if (!MI->getFlag(MachineInstr::FrameSetup) &&
            (PrologEndLoc.isUnknown() && !MI->getDebugLoc().isUnknown()))
          PrologEndLoc = MI->getDebugLoc();

        // Check if the instruction clobbers any registers with debug vars.
        for (MachineInstr::const_mop_iterator MOI = MI->operands_begin(),
               MOE = MI->operands_end(); MOI != MOE; ++MOI) {
          if (!MOI->isReg() || !MOI->isDef() || !MOI->getReg())
            continue;
          for (MCRegAliasIterator AI(MOI->getReg(), TRI, true);
               AI.isValid(); ++AI) {
            unsigned Reg = *AI;
            const MDNode *Var = LiveUserVar[Reg];
            if (!Var)
              continue;
            // Reg is now clobbered.
            LiveUserVar[Reg] = 0;

            // Was MD last defined by a DBG_VALUE referring to Reg?
            DbgValueHistoryMap::iterator HistI = DbgValues.find(Var);
            if (HistI == DbgValues.end())
              continue;
            SmallVectorImpl<const MachineInstr*> &History = HistI->second;
            if (History.empty())
              continue;
            const MachineInstr *Prev = History.back();
            // Sanity-check: Register assignments are terminated at the end of
            // their block.
            if (!Prev->isDebugValue() || Prev->getParent() != MI->getParent())
              continue;
            // Is the variable still in Reg?
            if (!isDbgValueInDefinedReg(Prev) ||
                Prev->getOperand(0).getReg() != Reg)
              continue;
            // Var is clobbered. Make sure the next instruction gets a label.
            History.push_back(MI);
          }
        }
      }
    }
  }

  for (DbgValueHistoryMap::iterator I = DbgValues.begin(), E = DbgValues.end();
       I != E; ++I) {
    SmallVectorImpl<const MachineInstr*> &History = I->second;
    if (History.empty())
      continue;

    // Make sure the final register assignments are terminated.
    const MachineInstr *Prev = History.back();
    if (Prev->isDebugValue() && isDbgValueInDefinedReg(Prev)) {
      const MachineBasicBlock *PrevMBB = Prev->getParent();
      MachineBasicBlock::const_iterator LastMI =
        PrevMBB->getLastNonDebugInstr();
      if (LastMI == PrevMBB->end())
        // Drop DBG_VALUE for empty range.
        History.pop_back();
      else {
        // Terminate after LastMI.
        History.push_back(LastMI);
      }
    }
    // Request labels for the full history.
    for (unsigned i = 0, e = History.size(); i != e; ++i) {
      const MachineInstr *MI = History[i];
      if (MI->isDebugValue())
        requestLabelBeforeInsn(MI);
      else
        requestLabelAfterInsn(MI);
    }
  }

  PrevInstLoc = DebugLoc();
  PrevLabel = FunctionBeginSym;

  // Record beginning of function.
  if (!PrologEndLoc.isUnknown()) {
    DebugLoc FnStartDL = getFnDebugLoc(PrologEndLoc,
                                       MF->getFunction()->getContext());
    recordSourceLine(FnStartDL.getLine(), FnStartDL.getCol(),
                     FnStartDL.getScope(MF->getFunction()->getContext()),
    // We'd like to list the prologue as "not statements" but GDB behaves
    // poorly if we do that. Revisit this with caution/GDB (7.5+) testing.
                     DWARF2_FLAG_IS_STMT);
  }
}

void DwarfDebug::addScopeVariable(LexicalScope *LS, DbgVariable *Var) {
  SmallVectorImpl<DbgVariable *> &Vars = ScopeVariables[LS];
  DIVariable DV = Var->getVariable();
  if (DV.getTag() == dwarf::DW_TAG_arg_variable) {
    DISubprogram Ctxt(DV.getContext());
    DIArray Variables = Ctxt.getVariables();
    // If the variable is a parameter (arg_variable) and this is an optimized
    // build (the subprogram has a 'variables' list) make sure we keep the
    // parameters in order. Otherwise we would produce an incorrect function
    // type with parameters out of order if function parameters were used out of
    // order or unused (see the call to addScopeVariable in endFunction where
    // the remaining unused variables (including parameters) are added).
    if (unsigned NumVariables = Variables.getNumElements()) {
      // Keep the parameters at the start of the variables list. Search through
      // current variable list (Vars) and the full function variable list in
      // lock-step looking for this parameter in the full list to find the
      // insertion point.
      SmallVectorImpl<DbgVariable *>::iterator I = Vars.begin();
      unsigned j = 0;
      while (I != Vars.end() && j != NumVariables &&
             Variables.getElement(j) != DV &&
             (*I)->getVariable().getTag() == dwarf::DW_TAG_arg_variable) {
        if (Variables.getElement(j) == (*I)->getVariable())
          ++I;
        ++j;
      }
      Vars.insert(I, Var);
      return;
    }
  }

  Vars.push_back(Var);
}

// Gather and emit post-function debug information.
void DwarfDebug::endFunction(const MachineFunction *MF) {
  if (!MMI->hasDebugInfo() || LScopes.empty()) return;

  // Define end label for subprogram.
  FunctionEndSym = Asm->GetTempSymbol("func_end",
                                      Asm->getFunctionNumber());
  // Assumes in correct section after the entry point.
  Asm->OutStreamer.EmitLabel(FunctionEndSym);
  // Set DwarfCompileUnitID in MCContext to default value.
  Asm->OutStreamer.getContext().setDwarfCompileUnitID(0);

  SmallPtrSet<const MDNode *, 16> ProcessedVars;
  collectVariableInfo(MF, ProcessedVars);

  LexicalScope *FnScope = LScopes.getCurrentFunctionScope();
  CompileUnit *TheCU = SPMap.lookup(FnScope->getScopeNode());
  assert(TheCU && "Unable to find compile unit!");

  // Construct abstract scopes.
  ArrayRef<LexicalScope *> AList = LScopes.getAbstractScopesList();
  for (unsigned i = 0, e = AList.size(); i != e; ++i) {
    LexicalScope *AScope = AList[i];
    DISubprogram SP(AScope->getScopeNode());
    if (SP.Verify()) {
      // Collect info for variables that were optimized out.
      DIArray Variables = SP.getVariables();
      for (unsigned i = 0, e = Variables.getNumElements(); i != e; ++i) {
        DIVariable DV(Variables.getElement(i));
        if (!DV || !DV.Verify() || !ProcessedVars.insert(DV))
          continue;
        // Check that DbgVariable for DV wasn't created earlier, when
        // findAbstractVariable() was called for inlined instance of DV.
        LLVMContext &Ctx = DV->getContext();
        DIVariable CleanDV = cleanseInlinedVariable(DV, Ctx);
        if (AbstractVariables.lookup(CleanDV))
          continue;
        if (LexicalScope *Scope = LScopes.findAbstractScope(DV.getContext()))
          addScopeVariable(Scope, new DbgVariable(DV, NULL));
      }
    }
    if (ProcessedSPNodes.count(AScope->getScopeNode()) == 0)
      constructScopeDIE(TheCU, AScope);
  }

  DIE *CurFnDIE = constructScopeDIE(TheCU, FnScope);

  if (!MF->getTarget().Options.DisableFramePointerElim(*MF))
    TheCU->addFlag(CurFnDIE, dwarf::DW_AT_APPLE_omit_frame_ptr);

  DebugFrames.push_back(FunctionDebugFrameInfo(Asm->getFunctionNumber(),
                                               MMI->getFrameMoves()));

  // Clear debug info
  for (DenseMap<LexicalScope *, SmallVector<DbgVariable *, 8> >::iterator
         I = ScopeVariables.begin(), E = ScopeVariables.end(); I != E; ++I)
    DeleteContainerPointers(I->second);
  ScopeVariables.clear();
  DeleteContainerPointers(CurrentFnArguments);
  UserVariables.clear();
  DbgValues.clear();
  AbstractVariables.clear();
  LabelsBeforeInsn.clear();
  LabelsAfterInsn.clear();
  PrevLabel = NULL;
}

// Register a source line with debug info. Returns the  unique label that was
// emitted and which provides correspondence to the source line list.
void DwarfDebug::recordSourceLine(unsigned Line, unsigned Col, const MDNode *S,
                                  unsigned Flags) {
  StringRef Fn;
  StringRef Dir;
  unsigned Src = 1;
  if (S) {
    DIDescriptor Scope(S);

    if (Scope.isCompileUnit()) {
      DICompileUnit CU(S);
      Fn = CU.getFilename();
      Dir = CU.getDirectory();
    } else if (Scope.isFile()) {
      DIFile F(S);
      Fn = F.getFilename();
      Dir = F.getDirectory();
    } else if (Scope.isSubprogram()) {
      DISubprogram SP(S);
      Fn = SP.getFilename();
      Dir = SP.getDirectory();
    } else if (Scope.isLexicalBlockFile()) {
      DILexicalBlockFile DBF(S);
      Fn = DBF.getFilename();
      Dir = DBF.getDirectory();
    } else if (Scope.isLexicalBlock()) {
      DILexicalBlock DB(S);
      Fn = DB.getFilename();
      Dir = DB.getDirectory();
    } else
      llvm_unreachable("Unexpected scope info");

    Src = getOrCreateSourceID(Fn, Dir,
            Asm->OutStreamer.getContext().getDwarfCompileUnitID());
  }
  Asm->OutStreamer.EmitDwarfLocDirective(Src, Line, Col, Flags, 0, 0, Fn);
}

//===----------------------------------------------------------------------===//
// Emit Methods
//===----------------------------------------------------------------------===//

// Compute the size and offset of a DIE.
unsigned
DwarfUnits::computeSizeAndOffset(DIE *Die, unsigned Offset) {
  // Get the children.
  const std::vector<DIE *> &Children = Die->getChildren();

  // Record the abbreviation.
  assignAbbrevNumber(Die->getAbbrev());

  // Get the abbreviation for this DIE.
  unsigned AbbrevNumber = Die->getAbbrevNumber();
  const DIEAbbrev *Abbrev = Abbreviations->at(AbbrevNumber - 1);

  // Set DIE offset
  Die->setOffset(Offset);

  // Start the size with the size of abbreviation code.
  Offset += MCAsmInfo::getULEB128Size(AbbrevNumber);

  const SmallVectorImpl<DIEValue*> &Values = Die->getValues();
  const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev->getData();

  // Size the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i)
    // Size attribute value.
    Offset += Values[i]->SizeOf(Asm, AbbrevData[i].getForm());

  // Size the DIE children if any.
  if (!Children.empty()) {
    assert(Abbrev->getChildrenFlag() == dwarf::DW_CHILDREN_yes &&
           "Children flag not set");

    for (unsigned j = 0, M = Children.size(); j < M; ++j)
      Offset = computeSizeAndOffset(Children[j], Offset);

    // End of children marker.
    Offset += sizeof(int8_t);
  }

  Die->setSize(Offset - Die->getOffset());
  return Offset;
}

// Compute the size and offset of all the DIEs.
void DwarfUnits::computeSizeAndOffsets() {
  // Offset from the beginning of debug info section.
  unsigned AccuOffset = 0;
  for (SmallVectorImpl<CompileUnit *>::iterator I = CUs.begin(),
         E = CUs.end(); I != E; ++I) {
    (*I)->setDebugInfoOffset(AccuOffset);
    unsigned Offset =
      sizeof(int32_t) + // Length of Compilation Unit Info
      sizeof(int16_t) + // DWARF version number
      sizeof(int32_t) + // Offset Into Abbrev. Section
      sizeof(int8_t);   // Pointer Size (in bytes)

    unsigned EndOffset = computeSizeAndOffset((*I)->getCUDie(), Offset);
    AccuOffset += EndOffset;
  }
}

// Emit initial Dwarf sections with a label at the start of each one.
void DwarfDebug::emitSectionLabels() {
  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();

  // Dwarf sections base addresses.
  DwarfInfoSectionSym =
    emitSectionSym(Asm, TLOF.getDwarfInfoSection(), "section_info");
  DwarfAbbrevSectionSym =
    emitSectionSym(Asm, TLOF.getDwarfAbbrevSection(), "section_abbrev");
  if (useSplitDwarf())
    DwarfAbbrevDWOSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfAbbrevDWOSection(),
                     "section_abbrev_dwo");
  emitSectionSym(Asm, TLOF.getDwarfARangesSection());

  if (const MCSection *MacroInfo = TLOF.getDwarfMacroInfoSection())
    emitSectionSym(Asm, MacroInfo);

  DwarfLineSectionSym =
    emitSectionSym(Asm, TLOF.getDwarfLineSection(), "section_line");
  emitSectionSym(Asm, TLOF.getDwarfLocSection());
  if (GenerateDwarfPubNamesSection)
    emitSectionSym(Asm, TLOF.getDwarfPubNamesSection());
  emitSectionSym(Asm, TLOF.getDwarfPubTypesSection());
  DwarfStrSectionSym =
    emitSectionSym(Asm, TLOF.getDwarfStrSection(), "info_string");
  if (useSplitDwarf()) {
    DwarfStrDWOSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfStrDWOSection(), "skel_string");
    DwarfAddrSectionSym =
      emitSectionSym(Asm, TLOF.getDwarfAddrSection(), "addr_sec");
  }
  DwarfDebugRangeSectionSym = emitSectionSym(Asm, TLOF.getDwarfRangesSection(),
                                             "debug_range");

  DwarfDebugLocSectionSym = emitSectionSym(Asm, TLOF.getDwarfLocSection(),
                                           "section_debug_loc");

  TextSectionSym = emitSectionSym(Asm, TLOF.getTextSection(), "text_begin");
  emitSectionSym(Asm, TLOF.getDataSection());
}

// Recursively emits a debug information entry.
void DwarfDebug::emitDIE(DIE *Die, std::vector<DIEAbbrev *> *Abbrevs) {
  // Get the abbreviation for this DIE.
  unsigned AbbrevNumber = Die->getAbbrevNumber();
  const DIEAbbrev *Abbrev = Abbrevs->at(AbbrevNumber - 1);

  // Emit the code (index) for the abbreviation.
  if (Asm->isVerbose())
    Asm->OutStreamer.AddComment("Abbrev [" + Twine(AbbrevNumber) + "] 0x" +
                                Twine::utohexstr(Die->getOffset()) + ":0x" +
                                Twine::utohexstr(Die->getSize()) + " " +
                                dwarf::TagString(Abbrev->getTag()));
  Asm->EmitULEB128(AbbrevNumber);

  const SmallVectorImpl<DIEValue*> &Values = Die->getValues();
  const SmallVectorImpl<DIEAbbrevData> &AbbrevData = Abbrev->getData();

  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; ++i) {
    unsigned Attr = AbbrevData[i].getAttribute();
    unsigned Form = AbbrevData[i].getForm();
    assert(Form && "Too many attributes for DIE (check abbreviation)");

    if (Asm->isVerbose())
      Asm->OutStreamer.AddComment(dwarf::AttributeString(Attr));

    switch (Attr) {
    case dwarf::DW_AT_abstract_origin: {
      DIEEntry *E = cast<DIEEntry>(Values[i]);
      DIE *Origin = E->getEntry();
      unsigned Addr = Origin->getOffset();
      if (Form == dwarf::DW_FORM_ref_addr) {
        // For DW_FORM_ref_addr, output the offset from beginning of debug info
        // section. Origin->getOffset() returns the offset from start of the
        // compile unit.
        DwarfUnits &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;
        Addr += Holder.getCUOffset(Origin->getCompileUnit());
      }
      Asm->EmitInt32(Addr);
      break;
    }
    case dwarf::DW_AT_ranges: {
      // DW_AT_range Value encodes offset in debug_range section.
      DIEInteger *V = cast<DIEInteger>(Values[i]);

      if (Asm->MAI->doesDwarfUseRelocationsAcrossSections()) {
        Asm->EmitLabelPlusOffset(DwarfDebugRangeSectionSym,
                                 V->getValue(),
                                 4);
      } else {
        Asm->EmitLabelOffsetDifference(DwarfDebugRangeSectionSym,
                                       V->getValue(),
                                       DwarfDebugRangeSectionSym,
                                       4);
      }
      break;
    }
    case dwarf::DW_AT_location: {
      if (DIELabel *L = dyn_cast<DIELabel>(Values[i])) {
        if (Asm->MAI->doesDwarfUseRelocationsAcrossSections())
          Asm->EmitLabelReference(L->getValue(), 4);
        else
          Asm->EmitLabelDifference(L->getValue(), DwarfDebugLocSectionSym, 4);
      } else {
        Values[i]->EmitValue(Asm, Form);
      }
      break;
    }
    case dwarf::DW_AT_accessibility: {
      if (Asm->isVerbose()) {
        DIEInteger *V = cast<DIEInteger>(Values[i]);
        Asm->OutStreamer.AddComment(dwarf::AccessibilityString(V->getValue()));
      }
      Values[i]->EmitValue(Asm, Form);
      break;
    }
    default:
      // Emit an attribute using the defined form.
      Values[i]->EmitValue(Asm, Form);
      break;
    }
  }

  // Emit the DIE children if any.
  if (Abbrev->getChildrenFlag() == dwarf::DW_CHILDREN_yes) {
    const std::vector<DIE *> &Children = Die->getChildren();

    for (unsigned j = 0, M = Children.size(); j < M; ++j)
      emitDIE(Children[j], Abbrevs);

    if (Asm->isVerbose())
      Asm->OutStreamer.AddComment("End Of Children Mark");
    Asm->EmitInt8(0);
  }
}

// Emit the various dwarf units to the unit section USection with
// the abbreviations going into ASection.
void DwarfUnits::emitUnits(DwarfDebug *DD,
                           const MCSection *USection,
                           const MCSection *ASection,
                           const MCSymbol *ASectionSym) {
  Asm->OutStreamer.SwitchSection(USection);
  for (SmallVectorImpl<CompileUnit *>::iterator I = CUs.begin(),
         E = CUs.end(); I != E; ++I) {
    CompileUnit *TheCU = *I;
    DIE *Die = TheCU->getCUDie();

    // Emit the compile units header.
    Asm->OutStreamer
      .EmitLabel(Asm->GetTempSymbol(USection->getLabelBeginName(),
                                    TheCU->getUniqueID()));

    // Emit size of content not including length itself
    unsigned ContentSize = Die->getSize() +
      sizeof(int16_t) + // DWARF version number
      sizeof(int32_t) + // Offset Into Abbrev. Section
      sizeof(int8_t);   // Pointer Size (in bytes)

    Asm->OutStreamer.AddComment("Length of Compilation Unit Info");
    Asm->EmitInt32(ContentSize);
    Asm->OutStreamer.AddComment("DWARF version number");
    Asm->EmitInt16(dwarf::DWARF_VERSION);
    Asm->OutStreamer.AddComment("Offset Into Abbrev. Section");
    Asm->EmitSectionOffset(Asm->GetTempSymbol(ASection->getLabelBeginName()),
                           ASectionSym);
    Asm->OutStreamer.AddComment("Address Size (in bytes)");
    Asm->EmitInt8(Asm->getDataLayout().getPointerSize());

    DD->emitDIE(Die, Abbreviations);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol(USection->getLabelEndName(),
                                                  TheCU->getUniqueID()));
  }
}

/// For a given compile unit DIE, returns offset from beginning of debug info.
unsigned DwarfUnits::getCUOffset(DIE *Die) {
  assert(Die->getTag() == dwarf::DW_TAG_compile_unit  &&
         "Input DIE should be compile unit in getCUOffset.");
  for (SmallVectorImpl<CompileUnit *>::iterator I = CUs.begin(),
       E = CUs.end(); I != E; ++I) {
    CompileUnit *TheCU = *I;
    if (TheCU->getCUDie() == Die)
      return TheCU->getDebugInfoOffset();
  }
  llvm_unreachable("The compile unit DIE should belong to CUs in DwarfUnits.");
}

// Emit the debug info section.
void DwarfDebug::emitDebugInfo() {
  DwarfUnits &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;

  Holder.emitUnits(this, Asm->getObjFileLowering().getDwarfInfoSection(),
                   Asm->getObjFileLowering().getDwarfAbbrevSection(),
                   DwarfAbbrevSectionSym);
}

// Emit the abbreviation section.
void DwarfDebug::emitAbbreviations() {
  if (!useSplitDwarf())
    emitAbbrevs(Asm->getObjFileLowering().getDwarfAbbrevSection(),
                &Abbreviations);
  else
    emitSkeletonAbbrevs(Asm->getObjFileLowering().getDwarfAbbrevSection());
}

void DwarfDebug::emitAbbrevs(const MCSection *Section,
                             std::vector<DIEAbbrev *> *Abbrevs) {
  // Check to see if it is worth the effort.
  if (!Abbrevs->empty()) {
    // Start the debug abbrev section.
    Asm->OutStreamer.SwitchSection(Section);

    MCSymbol *Begin = Asm->GetTempSymbol(Section->getLabelBeginName());
    Asm->OutStreamer.EmitLabel(Begin);

    // For each abbrevation.
    for (unsigned i = 0, N = Abbrevs->size(); i < N; ++i) {
      // Get abbreviation data
      const DIEAbbrev *Abbrev = Abbrevs->at(i);

      // Emit the abbrevations code (base 1 index.)
      Asm->EmitULEB128(Abbrev->getNumber(), "Abbreviation Code");

      // Emit the abbreviations data.
      Abbrev->Emit(Asm);
    }

    // Mark end of abbreviations.
    Asm->EmitULEB128(0, "EOM(3)");

    MCSymbol *End = Asm->GetTempSymbol(Section->getLabelEndName());
    Asm->OutStreamer.EmitLabel(End);
  }
}

// Emit the last address of the section and the end of the line matrix.
void DwarfDebug::emitEndOfLineMatrix(unsigned SectionEnd) {
  // Define last address of section.
  Asm->OutStreamer.AddComment("Extended Op");
  Asm->EmitInt8(0);

  Asm->OutStreamer.AddComment("Op size");
  Asm->EmitInt8(Asm->getDataLayout().getPointerSize() + 1);
  Asm->OutStreamer.AddComment("DW_LNE_set_address");
  Asm->EmitInt8(dwarf::DW_LNE_set_address);

  Asm->OutStreamer.AddComment("Section end label");

  Asm->OutStreamer.EmitSymbolValue(Asm->GetTempSymbol("section_end",SectionEnd),
                                   Asm->getDataLayout().getPointerSize());

  // Mark end of matrix.
  Asm->OutStreamer.AddComment("DW_LNE_end_sequence");
  Asm->EmitInt8(0);
  Asm->EmitInt8(1);
  Asm->EmitInt8(1);
}

// Emit visible names into a hashed accelerator table section.
void DwarfDebug::emitAccelNames() {
  DwarfAccelTable AT(DwarfAccelTable::Atom(DwarfAccelTable::eAtomTypeDIEOffset,
                                           dwarf::DW_FORM_data4));
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    const StringMap<std::vector<DIE*> > &Names = TheCU->getAccelNames();
    for (StringMap<std::vector<DIE*> >::const_iterator
           GI = Names.begin(), GE = Names.end(); GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      const std::vector<DIE *> &Entities = GI->second;
      for (std::vector<DIE *>::const_iterator DI = Entities.begin(),
             DE = Entities.end(); DI != DE; ++DI)
        AT.AddName(Name, (*DI));
    }
  }

  AT.FinalizeTable(Asm, "Names");
  Asm->OutStreamer.SwitchSection(
    Asm->getObjFileLowering().getDwarfAccelNamesSection());
  MCSymbol *SectionBegin = Asm->GetTempSymbol("names_begin");
  Asm->OutStreamer.EmitLabel(SectionBegin);

  // Emit the full data.
  AT.Emit(Asm, SectionBegin, &InfoHolder);
}

// Emit objective C classes and categories into a hashed accelerator table
// section.
void DwarfDebug::emitAccelObjC() {
  DwarfAccelTable AT(DwarfAccelTable::Atom(DwarfAccelTable::eAtomTypeDIEOffset,
                                           dwarf::DW_FORM_data4));
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    const StringMap<std::vector<DIE*> > &Names = TheCU->getAccelObjC();
    for (StringMap<std::vector<DIE*> >::const_iterator
           GI = Names.begin(), GE = Names.end(); GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      const std::vector<DIE *> &Entities = GI->second;
      for (std::vector<DIE *>::const_iterator DI = Entities.begin(),
             DE = Entities.end(); DI != DE; ++DI)
        AT.AddName(Name, (*DI));
    }
  }

  AT.FinalizeTable(Asm, "ObjC");
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering()
                                 .getDwarfAccelObjCSection());
  MCSymbol *SectionBegin = Asm->GetTempSymbol("objc_begin");
  Asm->OutStreamer.EmitLabel(SectionBegin);

  // Emit the full data.
  AT.Emit(Asm, SectionBegin, &InfoHolder);
}

// Emit namespace dies into a hashed accelerator table.
void DwarfDebug::emitAccelNamespaces() {
  DwarfAccelTable AT(DwarfAccelTable::Atom(DwarfAccelTable::eAtomTypeDIEOffset,
                                           dwarf::DW_FORM_data4));
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    const StringMap<std::vector<DIE*> > &Names = TheCU->getAccelNamespace();
    for (StringMap<std::vector<DIE*> >::const_iterator
           GI = Names.begin(), GE = Names.end(); GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      const std::vector<DIE *> &Entities = GI->second;
      for (std::vector<DIE *>::const_iterator DI = Entities.begin(),
             DE = Entities.end(); DI != DE; ++DI)
        AT.AddName(Name, (*DI));
    }
  }

  AT.FinalizeTable(Asm, "namespac");
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering()
                                 .getDwarfAccelNamespaceSection());
  MCSymbol *SectionBegin = Asm->GetTempSymbol("namespac_begin");
  Asm->OutStreamer.EmitLabel(SectionBegin);

  // Emit the full data.
  AT.Emit(Asm, SectionBegin, &InfoHolder);
}

// Emit type dies into a hashed accelerator table.
void DwarfDebug::emitAccelTypes() {
  std::vector<DwarfAccelTable::Atom> Atoms;
  Atoms.push_back(DwarfAccelTable::Atom(DwarfAccelTable::eAtomTypeDIEOffset,
                                        dwarf::DW_FORM_data4));
  Atoms.push_back(DwarfAccelTable::Atom(DwarfAccelTable::eAtomTypeTag,
                                        dwarf::DW_FORM_data2));
  Atoms.push_back(DwarfAccelTable::Atom(DwarfAccelTable::eAtomTypeTypeFlags,
                                        dwarf::DW_FORM_data1));
  DwarfAccelTable AT(Atoms);
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    const StringMap<std::vector<std::pair<DIE*, unsigned > > > &Names
      = TheCU->getAccelTypes();
    for (StringMap<std::vector<std::pair<DIE*, unsigned> > >::const_iterator
           GI = Names.begin(), GE = Names.end(); GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      const std::vector<std::pair<DIE *, unsigned> > &Entities = GI->second;
      for (std::vector<std::pair<DIE *, unsigned> >::const_iterator DI
             = Entities.begin(), DE = Entities.end(); DI !=DE; ++DI)
        AT.AddName(Name, (*DI).first, (*DI).second);
    }
  }

  AT.FinalizeTable(Asm, "types");
  Asm->OutStreamer.SwitchSection(Asm->getObjFileLowering()
                                 .getDwarfAccelTypesSection());
  MCSymbol *SectionBegin = Asm->GetTempSymbol("types_begin");
  Asm->OutStreamer.EmitLabel(SectionBegin);

  // Emit the full data.
  AT.Emit(Asm, SectionBegin, &InfoHolder);
}

/// emitDebugPubnames - Emit visible names into a debug pubnames section.
///
void DwarfDebug::emitDebugPubnames() {
  const MCSection *ISec = Asm->getObjFileLowering().getDwarfInfoSection();

  typedef DenseMap<const MDNode*, CompileUnit*> CUMapType;
  for (CUMapType::iterator I = CUMap.begin(), E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    unsigned ID = TheCU->getUniqueID();

    if (TheCU->getGlobalNames().empty())
      continue;

    // Start the dwarf pubnames section.
    Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfPubNamesSection());

    Asm->OutStreamer.AddComment("Length of Public Names Info");
    Asm->EmitLabelDifference(Asm->GetTempSymbol("pubnames_end", ID),
                             Asm->GetTempSymbol("pubnames_begin", ID), 4);

    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("pubnames_begin", ID));

    Asm->OutStreamer.AddComment("DWARF Version");
    Asm->EmitInt16(dwarf::DWARF_VERSION);

    Asm->OutStreamer.AddComment("Offset of Compilation Unit Info");
    Asm->EmitSectionOffset(Asm->GetTempSymbol(ISec->getLabelBeginName(), ID),
                           DwarfInfoSectionSym);

    Asm->OutStreamer.AddComment("Compilation Unit Length");
    Asm->EmitLabelDifference(Asm->GetTempSymbol(ISec->getLabelEndName(), ID),
                             Asm->GetTempSymbol(ISec->getLabelBeginName(), ID),
                             4);

    const StringMap<DIE*> &Globals = TheCU->getGlobalNames();
    for (StringMap<DIE*>::const_iterator
           GI = Globals.begin(), GE = Globals.end(); GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      const DIE *Entity = GI->second;

      Asm->OutStreamer.AddComment("DIE offset");
      Asm->EmitInt32(Entity->getOffset());

      if (Asm->isVerbose())
        Asm->OutStreamer.AddComment("External Name");
      Asm->OutStreamer.EmitBytes(StringRef(Name, strlen(Name)+1), 0);
    }

    Asm->OutStreamer.AddComment("End Mark");
    Asm->EmitInt32(0);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("pubnames_end", ID));
  }
}

void DwarfDebug::emitDebugPubTypes() {
  for (DenseMap<const MDNode *, CompileUnit *>::iterator I = CUMap.begin(),
         E = CUMap.end(); I != E; ++I) {
    CompileUnit *TheCU = I->second;
    // Start the dwarf pubtypes section.
    Asm->OutStreamer.SwitchSection(
      Asm->getObjFileLowering().getDwarfPubTypesSection());
    Asm->OutStreamer.AddComment("Length of Public Types Info");
    Asm->EmitLabelDifference(
      Asm->GetTempSymbol("pubtypes_end", TheCU->getUniqueID()),
      Asm->GetTempSymbol("pubtypes_begin", TheCU->getUniqueID()), 4);

    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("pubtypes_begin",
                                                  TheCU->getUniqueID()));

    if (Asm->isVerbose()) Asm->OutStreamer.AddComment("DWARF Version");
    Asm->EmitInt16(dwarf::DWARF_VERSION);

    Asm->OutStreamer.AddComment("Offset of Compilation Unit Info");
    const MCSection *ISec = Asm->getObjFileLowering().getDwarfInfoSection();
    Asm->EmitSectionOffset(Asm->GetTempSymbol(ISec->getLabelBeginName(),
                                              TheCU->getUniqueID()),
                           DwarfInfoSectionSym);

    Asm->OutStreamer.AddComment("Compilation Unit Length");
    Asm->EmitLabelDifference(Asm->GetTempSymbol(ISec->getLabelEndName(),
                                                TheCU->getUniqueID()),
                             Asm->GetTempSymbol(ISec->getLabelBeginName(),
                                                TheCU->getUniqueID()),
                             4);

    const StringMap<DIE*> &Globals = TheCU->getGlobalTypes();
    for (StringMap<DIE*>::const_iterator
           GI = Globals.begin(), GE = Globals.end(); GI != GE; ++GI) {
      const char *Name = GI->getKeyData();
      DIE *Entity = GI->second;

      if (Asm->isVerbose()) Asm->OutStreamer.AddComment("DIE offset");
      Asm->EmitInt32(Entity->getOffset());

      if (Asm->isVerbose()) Asm->OutStreamer.AddComment("External Name");
      // Emit the name with a terminating null byte.
      Asm->OutStreamer.EmitBytes(StringRef(Name, GI->getKeyLength()+1));
    }

    Asm->OutStreamer.AddComment("End Mark");
    Asm->EmitInt32(0);
    Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("pubtypes_end",
                                                  TheCU->getUniqueID()));
  }
}

// Emit strings into a string section.
void DwarfUnits::emitStrings(const MCSection *StrSection,
                             const MCSection *OffsetSection = NULL,
                             const MCSymbol *StrSecSym = NULL) {

  if (StringPool.empty()) return;

  // Start the dwarf str section.
  Asm->OutStreamer.SwitchSection(StrSection);

  // Get all of the string pool entries and put them in an array by their ID so
  // we can sort them.
  SmallVector<std::pair<unsigned,
                 StringMapEntry<std::pair<MCSymbol*, unsigned> >*>, 64> Entries;

  for (StringMap<std::pair<MCSymbol*, unsigned> >::iterator
         I = StringPool.begin(), E = StringPool.end();
       I != E; ++I)
    Entries.push_back(std::make_pair(I->second.second, &*I));

  array_pod_sort(Entries.begin(), Entries.end());

  for (unsigned i = 0, e = Entries.size(); i != e; ++i) {
    // Emit a label for reference from debug information entries.
    Asm->OutStreamer.EmitLabel(Entries[i].second->getValue().first);

    // Emit the string itself with a terminating null byte.
    Asm->OutStreamer.EmitBytes(StringRef(Entries[i].second->getKeyData(),
                                         Entries[i].second->getKeyLength()+1));
  }

  // If we've got an offset section go ahead and emit that now as well.
  if (OffsetSection) {
    Asm->OutStreamer.SwitchSection(OffsetSection);
    unsigned offset = 0;
    unsigned size = 4; // FIXME: DWARF64 is 8.
    for (unsigned i = 0, e = Entries.size(); i != e; ++i) {
      Asm->OutStreamer.EmitIntValue(offset, size);
      offset += Entries[i].second->getKeyLength() + 1;
    }
  }
}

// Emit strings into a string section.
void DwarfUnits::emitAddresses(const MCSection *AddrSection) {

  if (AddressPool.empty()) return;

  // Start the dwarf addr section.
  Asm->OutStreamer.SwitchSection(AddrSection);

  // Get all of the string pool entries and put them in an array by their ID so
  // we can sort them.
  SmallVector<std::pair<unsigned,
                        std::pair<MCSymbol*, unsigned>* >, 64> Entries;

  for (DenseMap<MCSymbol*, std::pair<MCSymbol*, unsigned> >::iterator
         I = AddressPool.begin(), E = AddressPool.end();
       I != E; ++I)
    Entries.push_back(std::make_pair(I->second.second, &(I->second)));

  array_pod_sort(Entries.begin(), Entries.end());

  for (unsigned i = 0, e = Entries.size(); i != e; ++i) {
    // Emit a label for reference from debug information entries.
    MCSymbol *Sym = Entries[i].second->first;
    if (Sym)
      Asm->EmitLabelReference(Entries[i].second->first,
                              Asm->getDataLayout().getPointerSize());
    else
      Asm->OutStreamer.EmitIntValue(0, Asm->getDataLayout().getPointerSize());
  }

}

// Emit visible names into a debug str section.
void DwarfDebug::emitDebugStr() {
  DwarfUnits &Holder = useSplitDwarf() ? SkeletonHolder : InfoHolder;
  Holder.emitStrings(Asm->getObjFileLowering().getDwarfStrSection());
}

// Emit visible names into a debug loc section.
void DwarfDebug::emitDebugLoc() {
  if (DotDebugLocEntries.empty())
    return;

  for (SmallVectorImpl<DotDebugLocEntry>::iterator
         I = DotDebugLocEntries.begin(), E = DotDebugLocEntries.end();
       I != E; ++I) {
    DotDebugLocEntry &Entry = *I;
    if (I + 1 != DotDebugLocEntries.end())
      Entry.Merge(I+1);
  }

  // Start the dwarf loc section.
  Asm->OutStreamer.SwitchSection(
    Asm->getObjFileLowering().getDwarfLocSection());
  unsigned char Size = Asm->getDataLayout().getPointerSize();
  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_loc", 0));
  unsigned index = 1;
  for (SmallVectorImpl<DotDebugLocEntry>::iterator
         I = DotDebugLocEntries.begin(), E = DotDebugLocEntries.end();
       I != E; ++I, ++index) {
    DotDebugLocEntry &Entry = *I;
    if (Entry.isMerged()) continue;
    if (Entry.isEmpty()) {
      Asm->OutStreamer.EmitIntValue(0, Size);
      Asm->OutStreamer.EmitIntValue(0, Size);
      Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_loc", index));
    } else {
      Asm->OutStreamer.EmitSymbolValue(Entry.Begin, Size);
      Asm->OutStreamer.EmitSymbolValue(Entry.End, Size);
      DIVariable DV(Entry.Variable);
      Asm->OutStreamer.AddComment("Loc expr size");
      MCSymbol *begin = Asm->OutStreamer.getContext().CreateTempSymbol();
      MCSymbol *end = Asm->OutStreamer.getContext().CreateTempSymbol();
      Asm->EmitLabelDifference(end, begin, 2);
      Asm->OutStreamer.EmitLabel(begin);
      if (Entry.isInt()) {
        DIBasicType BTy(DV.getType());
        if (BTy.Verify() &&
            (BTy.getEncoding()  == dwarf::DW_ATE_signed
             || BTy.getEncoding() == dwarf::DW_ATE_signed_char)) {
          Asm->OutStreamer.AddComment("DW_OP_consts");
          Asm->EmitInt8(dwarf::DW_OP_consts);
          Asm->EmitSLEB128(Entry.getInt());
        } else {
          Asm->OutStreamer.AddComment("DW_OP_constu");
          Asm->EmitInt8(dwarf::DW_OP_constu);
          Asm->EmitULEB128(Entry.getInt());
        }
      } else if (Entry.isLocation()) {
        if (!DV.hasComplexAddress())
          // Regular entry.
          Asm->EmitDwarfRegOp(Entry.Loc);
        else {
          // Complex address entry.
          unsigned N = DV.getNumAddrElements();
          unsigned i = 0;
          if (N >= 2 && DV.getAddrElement(0) == DIBuilder::OpPlus) {
            if (Entry.Loc.getOffset()) {
              i = 2;
              Asm->EmitDwarfRegOp(Entry.Loc);
              Asm->OutStreamer.AddComment("DW_OP_deref");
              Asm->EmitInt8(dwarf::DW_OP_deref);
              Asm->OutStreamer.AddComment("DW_OP_plus_uconst");
              Asm->EmitInt8(dwarf::DW_OP_plus_uconst);
              Asm->EmitSLEB128(DV.getAddrElement(1));
            } else {
              // If first address element is OpPlus then emit
              // DW_OP_breg + Offset instead of DW_OP_reg + Offset.
              MachineLocation Loc(Entry.Loc.getReg(), DV.getAddrElement(1));
              Asm->EmitDwarfRegOp(Loc);
              i = 2;
            }
          } else {
            Asm->EmitDwarfRegOp(Entry.Loc);
          }

          // Emit remaining complex address elements.
          for (; i < N; ++i) {
            uint64_t Element = DV.getAddrElement(i);
            if (Element == DIBuilder::OpPlus) {
              Asm->EmitInt8(dwarf::DW_OP_plus_uconst);
              Asm->EmitULEB128(DV.getAddrElement(++i));
            } else if (Element == DIBuilder::OpDeref) {
              if (!Entry.Loc.isReg())
                Asm->EmitInt8(dwarf::DW_OP_deref);
            } else
              llvm_unreachable("unknown Opcode found in complex address");
          }
        }
      }
      // else ... ignore constant fp. There is not any good way to
      // to represent them here in dwarf.
      Asm->OutStreamer.EmitLabel(end);
    }
  }
}

// Emit visible names into a debug aranges section.
void DwarfDebug::emitDebugARanges() {
  // Start the dwarf aranges section.
  Asm->OutStreamer.SwitchSection(
                          Asm->getObjFileLowering().getDwarfARangesSection());
}

// Emit visible names into a debug ranges section.
void DwarfDebug::emitDebugRanges() {
  // Start the dwarf ranges section.
  Asm->OutStreamer.SwitchSection(
    Asm->getObjFileLowering().getDwarfRangesSection());
  unsigned char Size = Asm->getDataLayout().getPointerSize();
  for (SmallVectorImpl<const MCSymbol *>::iterator
         I = DebugRangeSymbols.begin(), E = DebugRangeSymbols.end();
       I != E; ++I) {
    if (*I)
      Asm->OutStreamer.EmitSymbolValue(const_cast<MCSymbol*>(*I), Size);
    else
      Asm->OutStreamer.EmitIntValue(0, Size);
  }
}

// Emit visible names into a debug macinfo section.
void DwarfDebug::emitDebugMacInfo() {
  if (const MCSection *LineInfo =
      Asm->getObjFileLowering().getDwarfMacroInfoSection()) {
    // Start the dwarf macinfo section.
    Asm->OutStreamer.SwitchSection(LineInfo);
  }
}

// Emit inline info using following format.
// Section Header:
// 1. length of section
// 2. Dwarf version number
// 3. address size.
//
// Entries (one "entry" for each function that was inlined):
//
// 1. offset into __debug_str section for MIPS linkage name, if exists;
//   otherwise offset into __debug_str for regular function name.
// 2. offset into __debug_str section for regular function name.
// 3. an unsigned LEB128 number indicating the number of distinct inlining
// instances for the function.
//
// The rest of the entry consists of a {die_offset, low_pc} pair for each
// inlined instance; the die_offset points to the inlined_subroutine die in the
// __debug_info section, and the low_pc is the starting address for the
// inlining instance.
void DwarfDebug::emitDebugInlineInfo() {
  if (!Asm->MAI->doesDwarfUseInlineInfoSection())
    return;

  if (!FirstCU)
    return;

  Asm->OutStreamer.SwitchSection(
                        Asm->getObjFileLowering().getDwarfDebugInlineSection());

  Asm->OutStreamer.AddComment("Length of Debug Inlined Information Entry");
  Asm->EmitLabelDifference(Asm->GetTempSymbol("debug_inlined_end", 1),
                           Asm->GetTempSymbol("debug_inlined_begin", 1), 4);

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_inlined_begin", 1));

  Asm->OutStreamer.AddComment("Dwarf Version");
  Asm->EmitInt16(dwarf::DWARF_VERSION);
  Asm->OutStreamer.AddComment("Address Size (in bytes)");
  Asm->EmitInt8(Asm->getDataLayout().getPointerSize());

  for (SmallVectorImpl<const MDNode *>::iterator I = InlinedSPNodes.begin(),
         E = InlinedSPNodes.end(); I != E; ++I) {

    const MDNode *Node = *I;
    DenseMap<const MDNode *, SmallVector<InlineInfoLabels, 4> >::iterator II
      = InlineInfo.find(Node);
    SmallVectorImpl<InlineInfoLabels> &Labels = II->second;
    DISubprogram SP(Node);
    StringRef LName = SP.getLinkageName();
    StringRef Name = SP.getName();

    Asm->OutStreamer.AddComment("MIPS linkage name");
    if (LName.empty())
      Asm->EmitSectionOffset(InfoHolder.getStringPoolEntry(Name),
                             DwarfStrSectionSym);
    else
      Asm->EmitSectionOffset(InfoHolder
                             .getStringPoolEntry(getRealLinkageName(LName)),
                             DwarfStrSectionSym);

    Asm->OutStreamer.AddComment("Function name");
    Asm->EmitSectionOffset(InfoHolder.getStringPoolEntry(Name),
                           DwarfStrSectionSym);
    Asm->EmitULEB128(Labels.size(), "Inline count");

    for (SmallVectorImpl<InlineInfoLabels>::iterator LI = Labels.begin(),
           LE = Labels.end(); LI != LE; ++LI) {
      if (Asm->isVerbose()) Asm->OutStreamer.AddComment("DIE offset");
      Asm->EmitInt32(LI->second->getOffset());

      if (Asm->isVerbose()) Asm->OutStreamer.AddComment("low_pc");
      Asm->OutStreamer.EmitSymbolValue(LI->first,
                                       Asm->getDataLayout().getPointerSize());
    }
  }

  Asm->OutStreamer.EmitLabel(Asm->GetTempSymbol("debug_inlined_end", 1));
}

// DWARF5 Experimental Separate Dwarf emitters.

// This DIE has the following attributes: DW_AT_comp_dir, DW_AT_stmt_list,
// DW_AT_low_pc, DW_AT_high_pc, DW_AT_ranges, DW_AT_dwo_name, DW_AT_dwo_id,
// DW_AT_ranges_base, DW_AT_addr_base. If DW_AT_ranges is present,
// DW_AT_low_pc and DW_AT_high_pc are not used, and vice versa.
CompileUnit *DwarfDebug::constructSkeletonCU(const MDNode *N) {
  DICompileUnit DIUnit(N);
  CompilationDir = DIUnit.getDirectory();

  DIE *Die = new DIE(dwarf::DW_TAG_compile_unit);
  CompileUnit *NewCU = new CompileUnit(GlobalCUIndexCount++,
                                       DIUnit.getLanguage(), Die, Asm,
                                       this, &SkeletonHolder);

  NewCU->addLocalString(Die, dwarf::DW_AT_GNU_dwo_name,
                        DIUnit.getSplitDebugFilename());

  // This should be a unique identifier when we want to build .dwp files.
  NewCU->addUInt(Die, dwarf::DW_AT_GNU_dwo_id, dwarf::DW_FORM_data8, 0);

  // Relocate to the beginning of the addr_base section, else 0 for the
  // beginning of the one for this compile unit.
  if (Asm->MAI->doesDwarfUseRelocationsAcrossSections())
    NewCU->addLabel(Die, dwarf::DW_AT_GNU_addr_base, dwarf::DW_FORM_sec_offset,
                    DwarfAddrSectionSym);
  else
    NewCU->addUInt(Die, dwarf::DW_AT_GNU_addr_base,
                   dwarf::DW_FORM_sec_offset, 0);

  // 2.17.1 requires that we use DW_AT_low_pc for a single entry point
  // into an entity. We're using 0, or a NULL label for this.
  NewCU->addUInt(Die, dwarf::DW_AT_low_pc, dwarf::DW_FORM_addr, 0);

  // DW_AT_stmt_list is a offset of line number information for this
  // compile unit in debug_line section.
  // FIXME: Should handle multiple compile units.
  if (Asm->MAI->doesDwarfUseRelocationsAcrossSections())
    NewCU->addLabel(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_sec_offset,
                    DwarfLineSectionSym);
  else
    NewCU->addUInt(Die, dwarf::DW_AT_stmt_list, dwarf::DW_FORM_sec_offset, 0);

  if (!CompilationDir.empty())
    NewCU->addLocalString(Die, dwarf::DW_AT_comp_dir, CompilationDir);

  SkeletonHolder.addUnit(NewCU);
  SkeletonCUs.push_back(NewCU);

  return NewCU;
}

void DwarfDebug::emitSkeletonAbbrevs(const MCSection *Section) {
  assert(useSplitDwarf() && "No split dwarf debug info?");
  emitAbbrevs(Section, &SkeletonAbbrevs);
}

// Emit the .debug_info.dwo section for separated dwarf. This contains the
// compile units that would normally be in debug_info.
void DwarfDebug::emitDebugInfoDWO() {
  assert(useSplitDwarf() && "No split dwarf debug info?");
  InfoHolder.emitUnits(this, Asm->getObjFileLowering().getDwarfInfoDWOSection(),
                       Asm->getObjFileLowering().getDwarfAbbrevDWOSection(),
                       DwarfAbbrevDWOSectionSym);
}

// Emit the .debug_abbrev.dwo section for separated dwarf. This contains the
// abbreviations for the .debug_info.dwo section.
void DwarfDebug::emitDebugAbbrevDWO() {
  assert(useSplitDwarf() && "No split dwarf?");
  emitAbbrevs(Asm->getObjFileLowering().getDwarfAbbrevDWOSection(),
              &Abbreviations);
}

// Emit the .debug_str.dwo section for separated dwarf. This contains the
// string section and is identical in format to traditional .debug_str
// sections.
void DwarfDebug::emitDebugStrDWO() {
  assert(useSplitDwarf() && "No split dwarf?");
  const MCSection *OffSec = Asm->getObjFileLowering()
                            .getDwarfStrOffDWOSection();
  const MCSymbol *StrSym = DwarfStrSectionSym;
  InfoHolder.emitStrings(Asm->getObjFileLowering().getDwarfStrDWOSection(),
                         OffSec, StrSym);
}
