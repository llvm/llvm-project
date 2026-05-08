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

// Converts the file index of each line in the set to use our own internal
// file index. This is required for a reliable comparison as the DWARF index may
// differ across compilations.
static DenseSet<SourceLocation>
convertFileIndices(DenseSet<SourceLocation> Lines,
                   const DWARFDebugLine::LineTable *const LineTable,
                   DenseMap<uint16_t, uint16_t> &FileIndexMap,
                   StringMap<uint16_t> &FileNameMap) {
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
      if (NameIt != FileNameMap.end()) {
        Index = NameIt->second;
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
                        StringMap<uint16_t> &FileNameMap) {
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

  if (!Lines)
    return {};

  return convertFileIndices(Lines.value_or(DenseSet<SourceLocation>()),
                            LineTable, FileIndexMap, FileNameMap);
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
                          StringMap<uint16_t> &FileNameMap) {
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
                                     bool CombineInstances, raw_ostream &OS) {
  BaselineVarMap BaselineVars;
  StringMap<uint16_t> FileNameMap;

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

        auto Cov =
            computeVariableCoverage(VariableDIE, LT, FileIndexMap, FileNameMap);
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

      auto Cov =
          computeVariableCoverage(VariableDIE, LT, FileIndexMap, FileNameMap);
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
