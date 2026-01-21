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

/// Returns the set of source lines covered by a variable's debug information,
/// computed by intersecting the variable's location ranges and the containing
/// scope's address ranges.
static DenseSet<SourceLocation>
computeVariableCoverage(DWARFContext &DICtx, DWARFDie VariableDIE,
                        const DWARFDebugLine::LineTable *const LineTable) {
  /// Adds source locations to the set that correspond to an address range.
  auto addLines = [](const DWARFDebugLine::LineTable *LineTable,
                     DenseSet<SourceLocation> &Lines, DWARFAddressRange Range) {
    std::vector<uint32_t> Rows;
    if (LineTable->lookupAddressRange({Range.LowPC, Range.SectionIndex},
                                      Range.HighPC - Range.LowPC, Rows)) {
      for (const auto &RowI : Rows) {
        const auto Row = LineTable->Rows[RowI];
        // Lookup can return addresses below the LowPC - filter these out.
        if (Row.Address.Address < Range.LowPC)
          continue;
        const auto FileIndex = Row.File;

        const auto Line = Row.Line;
        if (Line) // Ignore zero lines.
          Lines.insert({FileIndex, Line});
      }
    }
  };

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
    Lines = ParentLines;
  else if (ParentLines)
    llvm::set_intersect(*Lines, *ParentLines);

  return Lines.value_or(DenseSet<SourceLocation>());
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
  size_t Instances;
};

typedef std::multimap<VarKey, VarCoverage, std::less<>> VarMap;

static std::optional<const VarKey> getVarKey(DWARFDie Die, DWARFDie Parent) {
  const auto *const DieName = Die.getName(DINameKind::LinkageName);
  const auto DieFile =
      Die.getDeclFile(DILineInfoSpecifier::FileLineInfoKind::RelativeFilePath);
  const auto *const ParentName = Parent.getName(DINameKind::LinkageName);
  if (!DieName || !ParentName)
    return std::nullopt;
  return VarKey{ParentName, DieName, DieFile, Die.getDeclLine()};
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
  OS << "\t" << Key.DeclFile << ":" << Key.DeclLine;
  OS << "\t" << format("%.3g", ((float)Var.Cov / Var.Instances));
  OS << "\n";
}

bool dwarfdump::showVariableCoverage(ObjectFile &Obj, DWARFContext &DICtx,
                                     bool CombineInstances, raw_ostream &OS) {
  VarMap Vars;

  for (const auto &U : DICtx.info_section_units()) {
    const auto *const LT = DICtx.getLineTableForUnit(U.get());
    for (const auto &Entry : U->dies()) {
      DWARFDie Die = {U.get(), &Entry};
      if (Die.getTag() != DW_TAG_variable &&
          Die.getTag() != DW_TAG_formal_parameter)
        continue;

      const auto Parents = getParentSubroutines(Die);
      if (!Parents.size())
        continue;
      const auto Parent = Parents.front();
      auto Key = getVarKey(Die, Parent);
      if (!Key)
        continue;

      const auto Cov = computeVariableCoverage(DICtx, Die, LT);

      VarCoverage VarCov = {Parents, Cov.size(), 1};

      Vars.insert({*Key, VarCov});
    }
  }

  std::pair<VarMap::iterator, VarMap::iterator> Range;

  OS << "\nVariable coverage statistics:\nFunction\t"
     << (CombineInstances ? "InstanceCount" : "InlChain")
     << "\tVariable\tDecl\tLinesCovered\n";

  if (CombineInstances) {
    for (auto FirstVar = Vars.begin(); FirstVar != Vars.end();
         FirstVar = Range.second) {
      Range = Vars.equal_range(FirstVar->first);
      VarCoverage CombinedCov = {{}, 0, 0};
      for (auto Var = Range.first; Var != Range.second; ++Var) {
        ++CombinedCov.Instances;
        CombinedCov.Cov += Var->second.Cov;
      }
      displayVariableCoverage(FirstVar->first, CombinedCov, true, OS);
    }
  } else {
    for (auto Var : Vars)
      displayVariableCoverage(Var.first, Var.second, false, OS);
  }

  return true;
}
