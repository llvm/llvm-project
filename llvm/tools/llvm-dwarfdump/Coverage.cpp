//===-- Coverage.cpp - Debug info coverage metrics ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-dwarfdump.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFAcceleratorTable.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace llvm::dwarf;
using namespace llvm::object;

/// Pair of file index and line number representing a source location.
typedef std::pair<uint16_t, size_t> SourceLocation;
/// Pair of subroutine name and variable name representing a local variable.
typedef std::pair<std::string, std::string> BitcodeVarKey;
/// Pair of file name and line number representing a source location.
typedef std::pair<StringRef, uint32_t> BitcodeSourceLocation;
/// Maps local variables found in the bitcode to a set of source locations.
typedef std::map<BitcodeVarKey, std::optional<DenseSet<BitcodeSourceLocation>>>
    BitcodeLineMap;

/// Adds source locations to the line set that correspond to an address range.
static void addLines(const DWARFDebugLine::LineTable *LineTable,
                     DenseSet<SourceLocation> &Lines, DWARFAddressRange Range) {
  std::vector<uint32_t> Rows;
  if (LineTable->lookupAddressRange({Range.LowPC, Range.SectionIndex},
                                    Range.HighPC - Range.LowPC, Rows)) {
    for (const auto &RowI : Rows) {
      const auto Row = LineTable->Rows[RowI];
      // Lookup can return addresses below the LowPC - filter these out.
      if (Row.Address.Address < Range.LowPC)
        continue;

      if (Row.Line) // Ignore zero lines.
        Lines.insert({Row.File, Row.Line});
    }
  }
}

/// Converts the file index of each line in the set to use our own internal
/// file index. This is required for a reliable comparison as the DWARF index
/// may differ across compilations.
static DenseSet<SourceLocation>
convertFileIndices(DenseSet<SourceLocation> Lines,
                   const DWARFDebugLine::LineTable *const LineTable,
                   DenseMap<uint16_t, uint16_t> &FileIndexMap,
                   StringMap<std::optional<uint16_t>> &FileNameMap) {
  DenseSet<SourceLocation> ResultLines;
  for (const auto &L : Lines) {
    uint16_t Index;
    const auto IndexIt = FileIndexMap.find(L.first);
    if (IndexIt != FileIndexMap.end()) {
      Index = IndexIt->second;
    } else {
      std::string Name;
      [[maybe_unused]] bool ValidIndex = LineTable->getFileNameByIndex(
          L.first, "", DILineInfoSpecifier::FileLineInfoKind::RelativeFilePath,
          Name);
      assert(ValidIndex && "File index was not valid for its own line table");

      auto NameIt = FileNameMap.find(Name);
      if (NameIt != FileNameMap.end() && NameIt->second) {
        Index = *NameIt->second;
      } else {
        Index = FileNameMap.size();
        FileNameMap.insert({Name, Index});
      }

      FileIndexMap.insert({L.first, Index});
    }

    ResultLines.insert({Index, L.second});
  }

  return ResultLines;
}

/// Returns the set of source lines covered by a variable's debug information,
/// computed by intersecting the variable's location ranges and the containing
/// scope's address ranges.
static DenseSet<SourceLocation>
computeVariableCoverage(DWARFDie VariableDIE,
                        const DWARFDebugLine::LineTable *const LineTable,
                        DenseMap<uint16_t, uint16_t> &FileIndexMap,
                        StringMap<std::optional<uint16_t>> &FileNameMap,
                        BitcodeLineMap::value_type *DefinedLines) {
  // The optionals below will be empty if no address ranges were found, and
  // present (but containing an empty set) if ranges were found but contained no
  // source locations, in order to distinguish the two cases.

  auto Locations = VariableDIE.getLocations(DW_AT_location);
  std::optional<DenseSet<SourceLocation>> Lines;
  if (Locations) {
    for (const auto &L : Locations.get()) {
      if (L.Range) {
        if (!Lines)
          Lines = DenseSet<SourceLocation>();
        addLines(LineTable, *Lines, L.Range.value());
      }
    }
  } else {
    // If the variable is optimized out and has no DW_AT_location, return an
    // empty set instead of falling back to the parent scope's address ranges.
    consumeError(Locations.takeError());
    return {};
  }

  // DW_AT_location attribute may contain overly broad address ranges, or none
  // at all, so we also consider the parent scope's address ranges if present.
  auto ParentRanges = VariableDIE.getParent().getAddressRanges();
  std::optional<DenseSet<SourceLocation>> ParentLines;
  if (ParentRanges) {
    ParentLines = DenseSet<SourceLocation>();
    for (const auto &R : ParentRanges.get())
      addLines(LineTable, *ParentLines, R);
  } else {
    consumeError(ParentRanges.takeError());
  }

  if (!Lines && ParentLines)
    Lines = std::move(ParentLines);
  else if (ParentLines)
    set_intersect(*Lines, *ParentLines);

  auto ResultLines =
      convertFileIndices(Lines.value_or(DenseSet<SourceLocation>()), LineTable,
                         FileIndexMap, FileNameMap);

  if (DefinedLines) {
    // Remove any lines where the variable does not have a defined value.
    auto &DL = DefinedLines->second;
    if (DL) {
      DenseSet<SourceLocation> IndexLines;
      for (const auto &L : *DL) {
        auto NameIt = FileNameMap.find(L.first);
        if (NameIt != FileNameMap.end()) {
          if (NameIt->second)
            IndexLines.insert({*NameIt->second, L.second});
        } else {
          // LineTable::getFileNameByIndex can return absolute paths even when
          // relative paths are requested, so search for keys that end with this
          // path as well.
          for (const auto &NameEntry : FileNameMap) {
            auto Name = NameEntry.first();
            if (Name.find(L.first, Name.size() - L.first.size()) !=
                std::string_view::npos) {
              FileNameMap.insert({L.first, NameEntry.second});
              IndexLines.insert({*NameEntry.second, L.second});
            }
          }
          if (FileNameMap.find(L.first) == FileNameMap.end()) {
            assert(0 && "Files found in bitcode but not in DWARF");
            FileNameMap.insert({L.first, std::nullopt});
          }
        }
      }
      if (!Lines)
        assert(0 && "Source lines found in bitcode but not in DWARF");
      else
        set_intersect(ResultLines, IndexLines);
    }
  }

  return ResultLines;
}

/// Adds source locations to the line set that are within an inlined subroutine.
static void getInlinedLines(DWARFDie SubroutineDIE,
                            DenseSet<SourceLocation> &Lines,
                            const DWARFDebugLine::LineTable *const LineTable) {
  for (const auto &ChildDIE : SubroutineDIE.children()) {
    if (ChildDIE.getTag() == DW_TAG_inlined_subroutine) {
      auto Ranges = ChildDIE.getAddressRanges();
      if (Ranges) {
        for (const auto &R : Ranges.get())
          addLines(LineTable, Lines, R);
      } else {
        consumeError(Ranges.takeError());
      }
    } else {
      getInlinedLines(ChildDIE, Lines, LineTable);
    }
  }
}

/// Returns the set of source lines present in the line table for a subroutine.
static DenseSet<SourceLocation>
computeSubroutineCoverage(DWARFDie SubroutineDIE,
                          const DWARFDebugLine::LineTable *const LineTable,
                          DenseMap<uint16_t, uint16_t> &FileIndexMap,
                          StringMap<std::optional<uint16_t>> &FileNameMap) {
  auto Ranges = SubroutineDIE.getAddressRanges();
  DenseSet<SourceLocation> Lines;
  if (Ranges) {
    for (const auto &R : Ranges.get())
      addLines(LineTable, Lines, R);
  } else {
    consumeError(Ranges.takeError());
  }

  // Exclude lines from any subroutines inlined into this one.
  DenseSet<SourceLocation> InlinedLines;
  getInlinedLines(SubroutineDIE, InlinedLines, LineTable);
  set_subtract(Lines, InlinedLines);

  return convertFileIndices(Lines, LineTable, FileIndexMap, FileNameMap);
}

static const SmallVector<DWARFDie> getParentSubroutines(DWARFDie DIE) {
  SmallVector<DWARFDie> Parents;
  DWARFDie Parent = DIE;
  do {
    if (Parent.getTag() == DW_TAG_subprogram) {
      Parents.push_back(Parent);
      break;
    }
    if (Parent.getTag() == DW_TAG_inlined_subroutine)
      Parents.push_back(Parent);
  } while ((Parent = Parent.getParent()));
  return Parents;
}

static bool isInScope(MDNode *Scope, const DebugLoc &Loc) {
  MDNode *Parent = Loc.getScope();
  while (Parent != Scope) {
    auto *S = dyn_cast_if_present<DIScope>(Parent);
    if (!S)
      return false;
    Parent = S->getScope();
  }
  return true;
}

/// Determines whether an instruction stores to a location. For the purposes of
/// this analysis, we consider any call-like instruction with the location as an
/// argument to be a store to it.
static bool isStoreToLocation(const DataLayout &DL, Instruction &I,
                              Value *Loc) {
  std::optional<at::AssignmentInfo> Info;
  if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
    if (SI->getPointerOperand() == Loc)
      return true;
    Info = at::getAssignmentInfo(DL, SI);
  } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&I)) {
    if (MI->getDest() == Loc)
      return true;
    Info = at::getAssignmentInfo(DL, MI);
  } else if (CallBase *CI = dyn_cast<CallBase>(&I)) {
    return CI->hasArgument(Loc);
  }
  return Info && Info->Base == Loc;
}

typedef SmallDenseMap<BasicBlock *, Instruction *, 8> VarDefinitionMap;

struct VarState {
  DbgVariableRecord &DVR;
  VarDefinitionMap Definitions;
};

/// Adds source locations to the line set for instructions in a basic block,
/// starting with a specific instruction.
static void addModuleLines(Instruction *I, VarState &Var,
                           DenseSet<std::pair<StringRef, uint32_t>> &Lines) {
  auto *VarScope = Var.DVR.getVariable()->getScope();
  do {
    auto &Loc = I->getDebugLoc();
    DIScope *Scope;
    if (Loc && isInScope(VarScope, Loc) && Loc.getLine() &&
        (Scope = dyn_cast_if_present<DIScope>(Loc.getScope()))) {
      Lines.insert({Scope->getFilename(), Loc.getLine()});
    }
  } while ((I = I->getNextNode()));
}

/// Computes the defined lines of all variables in an IR module.
static BitcodeLineMap processModule(Module *Mod) {
  BitcodeLineMap Result;
  std::vector<VarState> Vars;
  for (auto &F : Mod->functions()) {
    Vars.clear();
    for (auto &BB : F) {
      for (auto &I : BB) {
        for (DbgVariableRecord &DVR : filterDbgVars(I.getDbgRecordRange())) {
          if (DVR.isKillLocation()) {
            assert(0 && "Variable in bitcode has been optimized out");
            continue;
          }
          if (DVR.isDbgDeclare()) {
            // For #dbg_declare, don't treat the variable as live until we find
            // a store to it.
            Vars.push_back(VarState{DVR, VarDefinitionMap()});
          } else if (DVR.isDbgValue()) {
            // For #dbg_value, the variable is live immediately from this point.
            if (DVR.getDebugLoc().getInlinedAt() != nullptr) {
              assert(0 && "Variable in bitcode has been inlined");
              continue;
            }
            auto Var = find_if(Vars, [&](auto &Var) {
              return Var.DVR.getVariable() == DVR.getVariable();
            });
            if (Var != Vars.end())
              // If a basic block contains multiple stores to a variable, use
              // the earliest one by allowing the insertion to silently fail if
              // the basic block is already in the map.
              Var->Definitions.insert({&BB, &I});
            else
              Vars.push_back(VarState{DVR, {{&BB, &I}}});
          }
        }
      }
    }

    for (auto &BB : F) {
      for (auto &I : BB) {
        for (auto &Var : Vars) {
          if (isStoreToLocation(Mod->getDataLayout(), I, Var.DVR.getValue())) {
            // The variable is live from the instruction after the store. As
            // above, the earliest store in this basic block will be used.
            Var.Definitions.insert({&BB, I.getNextNode()});
          }
        }
      }
    }

    for (auto &Var : Vars) {
      SmallPtrSet<BasicBlock *, 8> Visited;
      DenseSet<std::pair<StringRef, uint32_t>> Lines;

      // Visit all basic blocks that are reachable from the entry block without
      // going through a block that stores to the variable.
      SmallVector<BasicBlock *> BlocksToVisit{&F.getEntryBlock()};
      while (!BlocksToVisit.empty()) {
        BasicBlock *BB = BlocksToVisit.pop_back_val();
        if (!Visited.insert(BB).second)
          continue;

        auto I = Var.Definitions.find(BB);
        if (I != Var.Definitions.end()) {
          // Block contains a definition: add all lines after it to the set
          if (I->second != nullptr)
            addModuleLines(I->second, Var, Lines);
        } else {
          // Block does not contain a definition: visit its successors
          auto S = successors(BB);
          BlocksToVisit.append(S.begin(), S.end());
        }
      }

      // All unvisited basic blocks must only be reachable by going through a
      // block that stores to the variable, so add lines to the set for all of
      // their instructions.
      for (auto &BB : F)
        if (!Visited.count(&BB))
          addModuleLines(&*BB.begin(), Var, Lines);

      BitcodeVarKey Key(F.getName(), Var.DVR.getVariable()->getName());
      Result.emplace(Key, Lines);
    }
  }
  return Result;
}

struct VarKey {
  const char *const SubprogramName;
  const char *const Name;
  std::string DeclFile;
  uint64_t DeclLine;

  bool operator==(const VarKey &Other) const {
    return DeclLine == Other.DeclLine &&
           !strcmp(SubprogramName, Other.SubprogramName) &&
           !strcmp(Name, Other.Name) && !DeclFile.compare(Other.DeclFile);
  }

  bool operator<(const VarKey &Other) const {
    int A = strcmp(SubprogramName, Other.SubprogramName);
    if (A)
      return A < 0;
    int B = strcmp(Name, Other.Name);
    if (B)
      return B < 0;
    int C = DeclFile.compare(Other.DeclFile);
    if (C)
      return C < 0;
    return DeclLine < Other.DeclLine;
  }
};

struct VarCoverage {
  SmallVector<DWARFDie> Parents;
  size_t Cov;
  size_t BaselineCov;
  size_t LTCov;
  size_t Missing;
  size_t Instances;
  bool MissingBaseline;
};

typedef std::multimap<VarKey, VarCoverage, std::less<>> VarMap;
typedef std::map<VarKey, DenseSet<SourceLocation>, std::less<>> BaselineVarMap;

static std::optional<const VarKey> getVarKey(DWARFDie VariableDIE,
                                             DWARFDie SubroutineDIE) {
  const auto *const VariableName = VariableDIE.getName(DINameKind::LinkageName);
  const auto DeclFile = VariableDIE.getDeclFile(
      DILineInfoSpecifier::FileLineInfoKind::RelativeFilePath);
  const auto *const SubroutineName =
      SubroutineDIE.getName(DINameKind::LinkageName);
  if (!VariableName || !SubroutineName)
    return std::nullopt;
  return VarKey{SubroutineName, VariableName, DeclFile,
                VariableDIE.getDeclLine()};
}

static void displayParents(SmallVector<DWARFDie> Parents, raw_ostream &OS) {
  bool First = true;
  for (const auto Parent : Parents) {
    if (auto FormValue = Parent.find(DW_AT_call_file)) {
      if (auto OptString = FormValue->getAsFile(
              DILineInfoSpecifier::FileLineInfoKind::RelativeFilePath)) {
        if (First)
          First = false;
        else
          OS << ", ";
        OS << *OptString << ":" << toUnsigned(Parent.find(DW_AT_call_line), 0);
      }
    }
  }
}

static void displayVariableCoverage(const VarKey &Key, const VarCoverage &Var,
                                    bool CombineInstances, raw_ostream &OS) {
  WithColor(OS, HighlightColor::String) << Key.SubprogramName;
  OS << "\t";
  if (CombineInstances)
    OS << Var.Instances;
  else if (Var.Parents.size())
    // FIXME: This may overflow the terminal if the inlining chain is large.
    displayParents(Var.Parents, OS);
  OS << "\t";
  WithColor(OS, HighlightColor::String) << Key.Name;
  OS << "\t";
  if (!Key.DeclFile.empty())
    OS << Key.DeclFile << ":" << Key.DeclLine;
  OS << "\t" << format("%.3g", ((float)Var.Cov / Var.Instances));
  if (Var.BaselineCov)
    OS << "\t" << format("%.3g", ((float)Var.BaselineCov / Var.Instances))
       << "\t" << format("%.3g", ((float)Var.Cov / Var.BaselineCov)) << "\t"
       << format("%.3g", ((float)Var.LTCov / Var.Instances)) << "\t"
       << format("%.3g", ((float)Var.LTCov / Var.BaselineCov));
  OS << "\n";
  if (Var.MissingBaseline)
    WithColor(errs(), HighlightColor::Warning).warning()
        << "DIE not found in baseline\n";
  if (Var.Missing)
    WithColor(errs(), HighlightColor::Warning).warning()
        << Var.Missing << " lines not found in baseline\n";
}

bool dwarfdump::showVariableCoverage(ObjectFile &Obj, DWARFContext &DICtx,
                                     ObjectFile *BaselineObj,
                                     DWARFContext *BaselineCtx,
                                     StringRef BitcodeFile,
                                     bool CombineInstances, raw_ostream &OS) {
  BitcodeLineMap LM;
  LLVMContext Context;
  if (!BitcodeFile.empty()) {
    SMDiagnostic Err;
    std::unique_ptr<Module> Mod = parseIRFile(BitcodeFile, Err, Context);
    if (!Err.getMessage().empty())
      Err.print("llvm-dwarfdump", OS);
    else
      LM = processModule(Mod.get());
  }

  BaselineVarMap BaselineVars;
  StringMap<std::optional<uint16_t>> FileNameMap;

  if (BaselineCtx) {
    for (const auto &U : BaselineCtx->info_section_units()) {
      const auto *const LT = BaselineCtx->getLineTableForUnit(U.get());
      DenseMap<uint16_t, uint16_t> FileIndexMap;
      for (const auto &Entry : U->dies()) {
        DWARFDie VariableDIE = {U.get(), &Entry};
        if (VariableDIE.getTag() != DW_TAG_variable &&
            VariableDIE.getTag() != DW_TAG_formal_parameter)
          continue;

        const auto Parents = getParentSubroutines(VariableDIE);
        if (!Parents.size())
          continue;
        const auto SubroutineDIE = Parents.front();
        auto Key = getVarKey(VariableDIE, SubroutineDIE);
        if (!Key)
          continue;

        const auto DefinedLines = LM.find({Key->SubprogramName, Key->Name});
        auto Cov = computeVariableCoverage(
            VariableDIE, LT, FileIndexMap, FileNameMap,
            DefinedLines != LM.end() ? &*DefinedLines : nullptr);
        const auto SubroutineCov = computeSubroutineCoverage(
            SubroutineDIE, LT, FileIndexMap, FileNameMap);
        set_intersect(Cov, SubroutineCov);

        auto Result = BaselineVars.insert({*Key, Cov});
        if (!Result.second)
          Result.first->second.insert_range(Cov);
      }
    }
  }

  VarMap Vars;

  for (const auto &U : DICtx.info_section_units()) {
    const auto *const LT = DICtx.getLineTableForUnit(U.get());
    DenseMap<uint16_t, uint16_t> FileIndexMap;
    for (const auto &Entry : U->dies()) {
      DWARFDie VariableDIE = {U.get(), &Entry};
      if (VariableDIE.getTag() != DW_TAG_variable &&
          VariableDIE.getTag() != DW_TAG_formal_parameter)
        continue;

      const auto Parents = getParentSubroutines(VariableDIE);
      if (!Parents.size())
        continue;
      const auto SubroutineDIE = Parents.front();
      auto Key = getVarKey(VariableDIE, SubroutineDIE);
      if (!Key)
        continue;

      const auto DefinedLines = LM.find({Key->SubprogramName, Key->Name});
      auto Cov = computeVariableCoverage(
          VariableDIE, LT, FileIndexMap, FileNameMap,
          DefinedLines != LM.end() ? &*DefinedLines : nullptr);
      const auto SubroutineCov = computeSubroutineCoverage(
          SubroutineDIE, LT, FileIndexMap, FileNameMap);
      set_intersect(Cov, SubroutineCov);

      VarCoverage VarCov = {Parents, Cov.size(), 0, 0, 0, 1, false};

      if (BaselineCtx) {
        BaselineVarMap::iterator Var = BaselineVars.find(*Key);

        if (Var != BaselineVars.end()) {
          const auto BCov = Var->second;
          VarCov.BaselineCov = BCov.size();

          for (const auto &L : Cov)
            VarCov.Missing += (1 - BCov.count(L));

          for (const auto &L : BCov)
            VarCov.LTCov += SubroutineCov.count(L);
        } else {
          VarCov.MissingBaseline = true;
        }
      }

      Vars.insert({*Key, VarCov});
    }
  }

  std::pair<VarMap::iterator, VarMap::iterator> Range;

  OS << "\nVariable coverage statistics:\nFunction\t"
     << (CombineInstances ? "InstanceCount" : "InlChain")
     << "\tVariable\tDecl\tLinesCovered";
  if (BaselineCtx)
    OS << "\tBaseline\tCoveredRatio\tLinesPresent\tLinesPresentRatio";
  OS << "\n";

  if (CombineInstances) {
    for (auto FirstVar = Vars.begin(); FirstVar != Vars.end();
         FirstVar = Range.second) {
      Range = Vars.equal_range(FirstVar->first);
      VarCoverage CombinedCov = {{}, 0, 0, 0, 0, 0, false};
      for (auto Var = Range.first; Var != Range.second; ++Var) {
        ++CombinedCov.Instances;
        CombinedCov.Cov += Var->second.Cov;
        CombinedCov.BaselineCov += Var->second.BaselineCov;
        CombinedCov.LTCov += Var->second.LTCov;
        CombinedCov.Missing += Var->second.Missing;
        CombinedCov.MissingBaseline |= Var->second.MissingBaseline;
      }
      displayVariableCoverage(FirstVar->first, CombinedCov, true, OS);
    }
  } else {
    for (auto Var : Vars)
      displayVariableCoverage(Var.first, Var.second, false, OS);
  }

  return true;
}
