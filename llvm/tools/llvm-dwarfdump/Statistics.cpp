//===-- Statistics.cpp - Debug Info quality metrics -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-dwarfdump.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "dwarfdump"
using namespace llvm;
using namespace llvm::dwarfdump;
using namespace llvm::object;

/// This represents the number of categories of debug location coverage being
/// calculated. The first category is the number of variables with 0% location
/// coverage, but the last category is the number of variables with 100%
/// location coverage.
constexpr int NumOfCoverageCategories = 12;

namespace {
/// Holds statistics for one function (or other entity that has a PC range and
/// contains variables, such as a compile unit).
struct PerFunctionStats {
  /// Number of inlined instances of this function.
  unsigned NumFnInlined = 0;
  /// Number of out-of-line instances of this function.
  unsigned NumFnOutOfLine = 0;
  /// Number of inlined instances that have abstract origins.
  unsigned NumAbstractOrigins = 0;
  /// Number of variables and parameters with location across all inlined
  /// instances.
  unsigned TotalVarWithLoc = 0;
  /// Number of constants with location across all inlined instances.
  unsigned ConstantMembers = 0;
  /// Number of arificial variables, parameters or members across all instances.
  unsigned NumArtificial = 0;
  /// List of all Variables and parameters in this function.
  StringSet<> VarsInFunction;
  /// Compile units also cover a PC range, but have this flag set to false.
  bool IsFunction = false;
  /// Function has source location information.
  bool HasSourceLocation = false;
  /// Number of function parameters.
  unsigned NumParams = 0;
  /// Number of function parameters with source location.
  unsigned NumParamSourceLocations = 0;
  /// Number of function parameters with type.
  unsigned NumParamTypes = 0;
  /// Number of function parameters with a DW_AT_location.
  unsigned NumParamLocations = 0;
  /// Number of local variables.
  unsigned NumLocalVars = 0;
  /// Number of local variables with source location.
  unsigned NumLocalVarSourceLocations = 0;
  /// Number of local variables with type.
  unsigned NumLocalVarTypes = 0;
  /// Number of local variables with DW_AT_location.
  unsigned NumLocalVarLocations = 0;
};

/// Holds accumulated global statistics about DIEs.
struct GlobalStats {
  /// Total number of PC range bytes covered by DW_AT_locations.
  unsigned ScopeBytesCovered = 0;
  /// Total number of PC range bytes in each variable's enclosing scope.
  unsigned ScopeBytes = 0;
  /// Total number of PC range bytes covered by DW_AT_locations with
  /// the debug entry values (DW_OP_entry_value).
  unsigned ScopeEntryValueBytesCovered = 0;
  /// Total number of PC range bytes covered by DW_AT_locations of
  /// formal parameters.
  unsigned ParamScopeBytesCovered = 0;
  /// Total number of PC range bytes in each parameter's enclosing scope.
  unsigned ParamScopeBytes = 0;
  /// Total number of PC range bytes covered by DW_AT_locations with
  /// the debug entry values (DW_OP_entry_value) (only for parameters).
  unsigned ParamScopeEntryValueBytesCovered = 0;
  /// Total number of PC range bytes covered by DW_AT_locations (only for local
  /// variables).
  unsigned LocalVarScopeBytesCovered = 0;
  /// Total number of PC range bytes in each local variable's enclosing scope.
  unsigned LocalVarScopeBytes = 0;
  /// Total number of PC range bytes covered by DW_AT_locations with
  /// the debug entry values (DW_OP_entry_value) (only for local variables).
  unsigned LocalVarScopeEntryValueBytesCovered = 0;
  /// Total number of call site entries (DW_AT_call_file & DW_AT_call_line).
  unsigned CallSiteEntries = 0;
  /// Total number of call site DIEs (DW_TAG_call_site).
  unsigned CallSiteDIEs = 0;
  /// Total number of call site parameter DIEs (DW_TAG_call_site_parameter).
  unsigned CallSiteParamDIEs = 0;
  /// Total byte size of concrete functions. This byte size includes
  /// inline functions contained in the concrete functions.
  unsigned FunctionSize = 0;
  /// Total byte size of inlined functions. This is the total number of bytes
  /// for the top inline functions within concrete functions. This can help
  /// tune the inline settings when compiling to match user expectations.
  unsigned InlineFunctionSize = 0;
};

/// Holds accumulated debug location statistics about local variables and
/// formal parameters.
struct LocationStats {
  /// Map the scope coverage decile to the number of variables in the decile.
  /// The first element of the array (at the index zero) represents the number
  /// of variables with the no debug location at all, but the last element
  /// in the vector represents the number of fully covered variables within
  /// its scope.
  std::vector<unsigned> VarParamLocStats{
      std::vector<unsigned>(NumOfCoverageCategories, 0)};
  /// Map non debug entry values coverage.
  std::vector<unsigned> VarParamNonEntryValLocStats{
      std::vector<unsigned>(NumOfCoverageCategories, 0)};
  /// The debug location statistics for formal parameters.
  std::vector<unsigned> ParamLocStats{
      std::vector<unsigned>(NumOfCoverageCategories, 0)};
  /// Map non debug entry values coverage for formal parameters.
  std::vector<unsigned> ParamNonEntryValLocStats{
      std::vector<unsigned>(NumOfCoverageCategories, 0)};
  /// The debug location statistics for local variables.
  std::vector<unsigned> LocalVarLocStats{
      std::vector<unsigned>(NumOfCoverageCategories, 0)};
  /// Map non debug entry values coverage for local variables.
  std::vector<unsigned> LocalVarNonEntryValLocStats{
      std::vector<unsigned>(NumOfCoverageCategories, 0)};
  /// Total number of local variables and function parameters processed.
  unsigned NumVarParam = 0;
  /// Total number of formal parameters processed.
  unsigned NumParam = 0;
  /// Total number of local variables processed.
  unsigned NumVar = 0;
};
} // namespace

/// Collect debug location statistics for one DIE.
static void collectLocStats(uint64_t BytesCovered, uint64_t BytesInScope,
                            std::vector<unsigned> &VarParamLocStats,
                            std::vector<unsigned> &ParamLocStats,
                            std::vector<unsigned> &LocalVarLocStats,
                            bool IsParam, bool IsLocalVar) {
  auto getCoverageBucket = [BytesCovered, BytesInScope]() -> unsigned {
    // No debug location at all for the variable.
    if (BytesCovered == 0)
      return 0;
    // Fully covered variable within its scope.
    if (BytesCovered >= BytesInScope)
      return NumOfCoverageCategories - 1;
    // Get covered range (e.g. 20%-29%).
    unsigned LocBucket = 100 * (double)BytesCovered / BytesInScope;
    LocBucket /= 10;
    return LocBucket + 1;
  };

  unsigned CoverageBucket = getCoverageBucket();
  VarParamLocStats[CoverageBucket]++;
  if (IsParam)
    ParamLocStats[CoverageBucket]++;
  else if (IsLocalVar)
    LocalVarLocStats[CoverageBucket]++;
}
/// Construct an identifier for a given DIE from its Prefix, Name, DeclFileName
/// and DeclLine. The identifier aims to be unique for any unique entities,
/// but keeping the same among different instances of the same entity.
static std::string constructDieID(DWARFDie Die,
                                  StringRef Prefix = StringRef()) {
  std::string IDStr;
  llvm::raw_string_ostream ID(IDStr);
  ID << Prefix
     << Die.getName(DINameKind::LinkageName);

  // Prefix + Name is enough for local variables and parameters.
  if (!Prefix.empty() && !Prefix.equals("g"))
    return ID.str();

  auto DeclFile = Die.findRecursively(dwarf::DW_AT_decl_file);
  std::string File;
  if (DeclFile) {
    DWARFUnit *U = Die.getDwarfUnit();
    if (const auto *LT = U->getContext().getLineTableForUnit(U))
      if (LT->getFileNameByIndex(
              dwarf::toUnsigned(DeclFile, 0), U->getCompilationDir(),
              DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath, File))
        File = std::string(sys::path::filename(File));
  }
  ID << ":" << (File.empty() ? "/" : File);
  ID << ":"
     << dwarf::toUnsigned(Die.findRecursively(dwarf::DW_AT_decl_line), 0);
  return ID.str();
}

/// Collect debug info quality metrics for one DIE.
static void collectStatsForDie(DWARFDie Die, std::string FnPrefix,
                               std::string VarPrefix, uint64_t BytesInScope,
                               uint32_t InlineDepth,
                               StringMap<PerFunctionStats> &FnStatMap,
                               GlobalStats &GlobalStats,
                               LocationStats &LocStats) {
  bool HasLoc = false;
  bool HasSrcLoc = false;
  bool HasType = false;
  uint64_t BytesCovered = 0;
  uint64_t BytesEntryValuesCovered = 0;
  auto &FnStats = FnStatMap[FnPrefix];
  bool IsParam = Die.getTag() == dwarf::DW_TAG_formal_parameter;
  bool IsLocalVar = Die.getTag() == dwarf::DW_TAG_variable;
  bool IsConstantMember = Die.getTag() == dwarf::DW_TAG_member &&
                          Die.find(dwarf::DW_AT_const_value);

  if (Die.getTag() == dwarf::DW_TAG_call_site ||
      Die.getTag() == dwarf::DW_TAG_GNU_call_site) {
    GlobalStats.CallSiteDIEs++;
    return;
  }

  if (Die.getTag() == dwarf::DW_TAG_call_site_parameter ||
      Die.getTag() == dwarf::DW_TAG_GNU_call_site_parameter) {
    GlobalStats.CallSiteParamDIEs++;
    return;
  }

  if (!IsParam && !IsLocalVar && !IsConstantMember) {
    // Not a variable or constant member.
    return;
  }

  // Ignore declarations of global variables.
  if (IsLocalVar && Die.find(dwarf::DW_AT_declaration))
    return;

  if (Die.findRecursively(dwarf::DW_AT_decl_file) &&
      Die.findRecursively(dwarf::DW_AT_decl_line))
    HasSrcLoc = true;

  if (Die.findRecursively(dwarf::DW_AT_type))
    HasType = true;

  auto IsEntryValue = [&](ArrayRef<uint8_t> D) -> bool {
    DWARFUnit *U = Die.getDwarfUnit();
    DataExtractor Data(toStringRef(D),
                       Die.getDwarfUnit()->getContext().isLittleEndian(), 0);
    DWARFExpression Expression(Data, U->getAddressByteSize(),
                               U->getFormParams().Format);
    // Consider the expression containing the DW_OP_entry_value as
    // an entry value.
    return llvm::any_of(Expression, [](DWARFExpression::Operation &Op) {
      return Op.getCode() == dwarf::DW_OP_entry_value ||
             Op.getCode() == dwarf::DW_OP_GNU_entry_value;
    });
  };

  if (Die.find(dwarf::DW_AT_const_value)) {
    // This catches constant members *and* variables.
    HasLoc = true;
    BytesCovered = BytesInScope;
  } else {
    // Handle variables and function arguments.
    Expected<std::vector<DWARFLocationExpression>> Loc =
        Die.getLocations(dwarf::DW_AT_location);
    if (!Loc) {
      consumeError(Loc.takeError());
    } else {
      HasLoc = true;
      // Get PC coverage.
      auto Default = find_if(
          *Loc, [](const DWARFLocationExpression &L) { return !L.Range; });
      if (Default != Loc->end()) {
        // Assume the entire range is covered by a single location.
        BytesCovered = BytesInScope;
      } else {
        for (auto Entry : *Loc) {
          uint64_t BytesEntryCovered = Entry.Range->HighPC - Entry.Range->LowPC;
          BytesCovered += BytesEntryCovered;
          if (IsEntryValue(Entry.Expr))
            BytesEntryValuesCovered += BytesEntryCovered;
        }
      }
    }
  }

  // Calculate the debug location statistics.
  if (BytesInScope) {
    LocStats.NumVarParam++;
    if (IsParam)
      LocStats.NumParam++;
    else if (IsLocalVar)
      LocStats.NumVar++;

    collectLocStats(BytesCovered, BytesInScope, LocStats.VarParamLocStats,
                    LocStats.ParamLocStats, LocStats.LocalVarLocStats, IsParam,
                    IsLocalVar);
    // Non debug entry values coverage statistics.
    collectLocStats(BytesCovered - BytesEntryValuesCovered, BytesInScope,
                    LocStats.VarParamNonEntryValLocStats,
                    LocStats.ParamNonEntryValLocStats,
                    LocStats.LocalVarNonEntryValLocStats, IsParam, IsLocalVar);
  }

  // Collect PC range coverage data.
  if (DWARFDie D =
          Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_abstract_origin))
    Die = D;

  std::string VarID = constructDieID(Die, VarPrefix);
  FnStats.VarsInFunction.insert(VarID);

  if (BytesInScope) {
    // Turns out we have a lot of ranges that extend past the lexical scope.
    GlobalStats.ScopeBytesCovered += std::min(BytesInScope, BytesCovered);
    GlobalStats.ScopeBytes += BytesInScope;
    GlobalStats.ScopeEntryValueBytesCovered += BytesEntryValuesCovered;
    if (IsParam) {
      GlobalStats.ParamScopeBytesCovered +=
          std::min(BytesInScope, BytesCovered);
      GlobalStats.ParamScopeBytes += BytesInScope;
      GlobalStats.ParamScopeEntryValueBytesCovered += BytesEntryValuesCovered;
    } else if (IsLocalVar) {
      GlobalStats.LocalVarScopeBytesCovered +=
          std::min(BytesInScope, BytesCovered);
      GlobalStats.LocalVarScopeBytes += BytesInScope;
      GlobalStats.LocalVarScopeEntryValueBytesCovered +=
          BytesEntryValuesCovered;
    }
    assert(GlobalStats.ScopeBytesCovered <= GlobalStats.ScopeBytes);
  }

  if (IsConstantMember) {
    FnStats.ConstantMembers++;
    return;
  }

  FnStats.TotalVarWithLoc += (unsigned)HasLoc;

  if (Die.find(dwarf::DW_AT_artificial)) {
    FnStats.NumArtificial++;
    return;
  }

  if (IsParam) {
    FnStats.NumParams++;
    if (HasType)
      FnStats.NumParamTypes++;
    if (HasSrcLoc)
      FnStats.NumParamSourceLocations++;
    if (HasLoc)
      FnStats.NumParamLocations++;
  } else if (IsLocalVar) {
    FnStats.NumLocalVars++;
    if (HasType)
      FnStats.NumLocalVarTypes++;
    if (HasSrcLoc)
      FnStats.NumLocalVarSourceLocations++;
    if (HasLoc)
      FnStats.NumLocalVarLocations++;
  }
}

/// Recursively collect debug info quality metrics.
static void collectStatsRecursive(DWARFDie Die, std::string FnPrefix,
                                  std::string VarPrefix, uint64_t BytesInScope,
                                  uint32_t InlineDepth,
                                  StringMap<PerFunctionStats> &FnStatMap,
                                  GlobalStats &GlobalStats,
                                  LocationStats &LocStats) {
  const dwarf::Tag Tag = Die.getTag();
  // Skip function types.
  if (Tag == dwarf::DW_TAG_subroutine_type)
    return;

  // Handle any kind of lexical scope.
  const bool IsFunction = Tag == dwarf::DW_TAG_subprogram;
  const bool IsBlock = Tag == dwarf::DW_TAG_lexical_block;
  const bool IsInlinedFunction = Tag == dwarf::DW_TAG_inlined_subroutine;
  if (IsFunction || IsInlinedFunction || IsBlock) {

    // Reset VarPrefix when entering a new function.
    if (Die.getTag() == dwarf::DW_TAG_subprogram ||
        Die.getTag() == dwarf::DW_TAG_inlined_subroutine)
      VarPrefix = "v";

    // Ignore forward declarations.
    if (Die.find(dwarf::DW_AT_declaration))
      return;

    // Check for call sites.
    if (Die.find(dwarf::DW_AT_call_file) && Die.find(dwarf::DW_AT_call_line))
      GlobalStats.CallSiteEntries++;

    // PC Ranges.
    auto RangesOrError = Die.getAddressRanges();
    if (!RangesOrError) {
      llvm::consumeError(RangesOrError.takeError());
      return;
    }

    auto Ranges = RangesOrError.get();
    uint64_t BytesInThisScope = 0;
    for (auto Range : Ranges)
      BytesInThisScope += Range.HighPC - Range.LowPC;

    // Count the function.
    if (!IsBlock) {
      // Skip over abstract origins.
      if (Die.find(dwarf::DW_AT_inline))
        return;
      std::string FnID = constructDieID(Die);
      // We've seen an instance of this function.
      auto &FnStats = FnStatMap[FnID];
      FnStats.IsFunction = true;
      if (IsInlinedFunction) {
        FnStats.NumFnInlined++;
        if (Die.findRecursively(dwarf::DW_AT_abstract_origin))
          FnStats.NumAbstractOrigins++;
      } else {
        FnStats.NumFnOutOfLine++;
      }
      if (Die.findRecursively(dwarf::DW_AT_decl_file) &&
          Die.findRecursively(dwarf::DW_AT_decl_line))
        FnStats.HasSourceLocation = true;
      // Update function prefix.
      FnPrefix = FnID;
    }

    if (BytesInThisScope) {
      BytesInScope = BytesInThisScope;
      if (IsFunction)
        GlobalStats.FunctionSize += BytesInThisScope;
      else if (IsInlinedFunction && InlineDepth == 0)
        GlobalStats.InlineFunctionSize += BytesInThisScope;
    }
  } else {
    // Not a scope, visit the Die itself. It could be a variable.
    collectStatsForDie(Die, FnPrefix, VarPrefix, BytesInScope, InlineDepth,
                       FnStatMap, GlobalStats, LocStats);
  }

  // Set InlineDepth correctly for child recursion
  if (IsFunction)
    InlineDepth = 0;
  else if (IsInlinedFunction)
    ++InlineDepth;

  // Traverse children.
  unsigned LexicalBlockIndex = 0;
  unsigned FormalParameterIndex = 0;
  DWARFDie Child = Die.getFirstChild();
  while (Child) {
    std::string ChildVarPrefix = VarPrefix;
    if (Child.getTag() == dwarf::DW_TAG_lexical_block)
      ChildVarPrefix += toHex(LexicalBlockIndex++) + '.';
    if (Child.getTag() == dwarf::DW_TAG_formal_parameter)
      ChildVarPrefix += 'p' + toHex(FormalParameterIndex++) + '.';

    collectStatsRecursive(Child, FnPrefix, ChildVarPrefix, BytesInScope,
                          InlineDepth, FnStatMap, GlobalStats, LocStats);
    Child = Child.getSibling();
  }
}

/// Print machine-readable output.
/// The machine-readable format is single-line JSON output.
/// \{
static void printDatum(raw_ostream &OS, const char *Key, json::Value Value) {
  OS << ",\"" << Key << "\":" << Value;
  LLVM_DEBUG(llvm::dbgs() << Key << ": " << Value << '\n');
}

static void printLocationStats(raw_ostream &OS, const char *Key,
                               std::vector<unsigned> &LocationStats) {
  OS << ",\"" << Key << " with 0% of parent scope covered by DW_AT_location\":"
     << LocationStats[0];
  LLVM_DEBUG(
      llvm::dbgs() << Key
                   << " with 0% of parent scope covered by DW_AT_location: \\"
                   << LocationStats[0] << '\n');
  OS << ",\"" << Key
     << " with (0%,10%) of parent scope covered by DW_AT_location\":"
     << LocationStats[1];
  LLVM_DEBUG(llvm::dbgs()
             << Key
             << " with (0%,10%) of parent scope covered by DW_AT_location: "
             << LocationStats[1] << '\n');
  for (unsigned i = 2; i < NumOfCoverageCategories - 1; ++i) {
    OS << ",\"" << Key << " with [" << (i - 1) * 10 << "%," << i * 10
       << "%) of parent scope covered by DW_AT_location\":" << LocationStats[i];
    LLVM_DEBUG(llvm::dbgs()
               << Key << " with [" << (i - 1) * 10 << "%," << i * 10
               << "%) of parent scope covered by DW_AT_location: "
               << LocationStats[i]);
  }
  OS << ",\"" << Key
     << " with 100% of parent scope covered by DW_AT_location\":"
     << LocationStats[NumOfCoverageCategories - 1];
  LLVM_DEBUG(
      llvm::dbgs() << Key
                   << " with 100% of parent scope covered by DW_AT_location: "
                   << LocationStats[NumOfCoverageCategories - 1]);
}

static void printSectionSizes(raw_ostream &OS, const SectionSizes &Sizes) {
  for (const auto &DebugSec : Sizes.DebugSectionSizes)
    OS << ",\"#bytes in " << DebugSec.getKey() << "\":" << DebugSec.getValue();
}

/// \}

/// Collect debug info quality metrics for an entire DIContext.
///
/// Do the impossible and reduce the quality of the debug info down to a few
/// numbers. The idea is to condense the data into numbers that can be tracked
/// over time to identify trends in newer compiler versions and gauge the effect
/// of particular optimizations. The raw numbers themselves are not particularly
/// useful, only the delta between compiling the same program with different
/// compilers is.
bool dwarfdump::collectStatsForObjectFile(ObjectFile &Obj, DWARFContext &DICtx,
                                          const Twine &Filename,
                                          raw_ostream &OS) {
  StringRef FormatName = Obj.getFileFormatName();
  GlobalStats GlobalStats;
  LocationStats LocStats;
  StringMap<PerFunctionStats> Statistics;
  for (const auto &CU : static_cast<DWARFContext *>(&DICtx)->compile_units())
    if (DWARFDie CUDie = CU->getNonSkeletonUnitDIE(false))
      collectStatsRecursive(CUDie, "/", "g", 0, 0, Statistics, GlobalStats,
                            LocStats);

  /// Collect the sizes of debug sections.
  SectionSizes Sizes;
  calculateSectionSizes(Obj, Sizes, Filename);

  /// The version number should be increased every time the algorithm is changed
  /// (including bug fixes). New metrics may be added without increasing the
  /// version.
  unsigned Version = 5;
  unsigned VarParamTotal = 0;
  unsigned VarParamUnique = 0;
  unsigned VarParamWithLoc = 0;
  unsigned NumFunctions = 0;
  unsigned NumInlinedFunctions = 0;
  unsigned NumFuncsWithSrcLoc = 0;
  unsigned NumAbstractOrigins = 0;
  unsigned ParamTotal = 0;
  unsigned ParamWithType = 0;
  unsigned ParamWithLoc = 0;
  unsigned ParamWithSrcLoc = 0;
  unsigned LocalVarTotal = 0;
  unsigned LocalVarWithType = 0;
  unsigned LocalVarWithSrcLoc = 0;
  unsigned LocalVarWithLoc = 0;
  for (auto &Entry : Statistics) {
    PerFunctionStats &Stats = Entry.getValue();
    unsigned TotalVars = Stats.VarsInFunction.size() *
                         (Stats.NumFnInlined + Stats.NumFnOutOfLine);
    // Count variables in global scope.
    if (!Stats.IsFunction)
      TotalVars =
          Stats.NumLocalVars + Stats.ConstantMembers + Stats.NumArtificial;
    unsigned Constants = Stats.ConstantMembers;
    VarParamWithLoc += Stats.TotalVarWithLoc + Constants;
    VarParamTotal += TotalVars;
    VarParamUnique += Stats.VarsInFunction.size();
    LLVM_DEBUG(for (auto &V
                    : Stats.VarsInFunction) llvm::dbgs()
               << Entry.getKey() << ": " << V.getKey() << "\n");
    NumFunctions += Stats.IsFunction;
    NumFuncsWithSrcLoc += Stats.HasSourceLocation;
    NumInlinedFunctions += Stats.IsFunction * Stats.NumFnInlined;
    NumAbstractOrigins += Stats.IsFunction * Stats.NumAbstractOrigins;
    ParamTotal += Stats.NumParams;
    ParamWithType += Stats.NumParamTypes;
    ParamWithLoc += Stats.NumParamLocations;
    ParamWithSrcLoc += Stats.NumParamSourceLocations;
    LocalVarTotal += Stats.NumLocalVars;
    LocalVarWithType += Stats.NumLocalVarTypes;
    LocalVarWithLoc += Stats.NumLocalVarLocations;
    LocalVarWithSrcLoc += Stats.NumLocalVarSourceLocations;
  }

  // Print summary.
  OS.SetBufferSize(1024);
  OS << "{\"version\":" << Version;
  LLVM_DEBUG(llvm::dbgs() << "Variable location quality metrics\n";
             llvm::dbgs() << "---------------------------------\n");

  printDatum(OS, "file", Filename.str());
  printDatum(OS, "format", FormatName);

  printDatum(OS, "#functions", NumFunctions);
  printDatum(OS, "#functions with location", NumFuncsWithSrcLoc);
  printDatum(OS, "#inlined functions", NumInlinedFunctions);
  printDatum(OS, "#inlined functions with abstract origins",
             NumAbstractOrigins);

  // This includes local variables and formal parameters.
  printDatum(OS, "#unique source variables", VarParamUnique);
  printDatum(OS, "#source variables", VarParamTotal);
  printDatum(OS, "#source variables with location", VarParamWithLoc);

  printDatum(OS, "#call site entries", GlobalStats.CallSiteEntries);
  printDatum(OS, "#call site DIEs", GlobalStats.CallSiteDIEs);
  printDatum(OS, "#call site parameter DIEs", GlobalStats.CallSiteParamDIEs);

  printDatum(OS, "sum_all_variables(#bytes in parent scope)",
             GlobalStats.ScopeBytes);
  printDatum(OS,
             "sum_all_variables(#bytes in parent scope covered by "
             "DW_AT_location)",
             GlobalStats.ScopeBytesCovered);
  printDatum(OS,
             "sum_all_variables(#bytes in parent scope covered by "
             "DW_OP_entry_value)",
             GlobalStats.ScopeEntryValueBytesCovered);

  printDatum(OS, "sum_all_params(#bytes in parent scope)",
             GlobalStats.ParamScopeBytes);
  printDatum(
      OS,
      "sum_all_params(#bytes in parent scope covered by DW_AT_location)",
      GlobalStats.ParamScopeBytesCovered);
  printDatum(OS,
             "sum_all_params(#bytes in parent scope covered by "
             "DW_OP_entry_value)",
             GlobalStats.ParamScopeEntryValueBytesCovered);

  printDatum(OS, "sum_all_local_vars(#bytes in parent scope)",
             GlobalStats.LocalVarScopeBytes);
  printDatum(OS,
             "sum_all_local_vars(#bytes in parent scope covered by "
             "DW_AT_location)",
             GlobalStats.LocalVarScopeBytesCovered);
  printDatum(OS,
             "sum_all_local_vars(#bytes in parent scope covered by "
             "DW_OP_entry_value)",
             GlobalStats.LocalVarScopeEntryValueBytesCovered);

  printDatum(OS, "#bytes witin functions", GlobalStats.FunctionSize);
  printDatum(OS, "#bytes witin inlined functions",
             GlobalStats.InlineFunctionSize);

  // Print the summary for formal parameters.
  printDatum(OS, "#params", ParamTotal);
  printDatum(OS, "#params with source location", ParamWithSrcLoc);
  printDatum(OS, "#params with type", ParamWithType);
  printDatum(OS, "#params with binary location", ParamWithLoc);

  // Print the summary for local variables.
  printDatum(OS, "#local vars", LocalVarTotal);
  printDatum(OS, "#local vars with source location", LocalVarWithSrcLoc);
  printDatum(OS, "#local vars with type", LocalVarWithType);
  printDatum(OS, "#local vars with binary location", LocalVarWithLoc);

  // Print the debug section sizes.
  printSectionSizes(OS, Sizes);

  // Print the location statistics for variables (includes local variables
  // and formal parameters).
  printDatum(OS, "#variables processed by location statistics",
             LocStats.NumVarParam);
  printLocationStats(OS, "#variables", LocStats.VarParamLocStats);
  printLocationStats(OS, "#variables - entry values",
                     LocStats.VarParamNonEntryValLocStats);

  // Print the location statistics for formal parameters.
  printDatum(OS, "#params processed by location statistics", LocStats.NumParam);
  printLocationStats(OS, "#params", LocStats.ParamLocStats);
  printLocationStats(OS, "#params - entry values",
                     LocStats.ParamNonEntryValLocStats);

  // Print the location statistics for local variables.
  printDatum(OS, "#local vars processed by location statistics",
             LocStats.NumVar);
  printLocationStats(OS, "#local vars", LocStats.LocalVarLocStats);
  printLocationStats(OS, "#local vars - entry values",
                     LocStats.LocalVarNonEntryValLocStats);
  OS << "}\n";
  LLVM_DEBUG(
      llvm::dbgs() << "Total Availability: "
                   << (int)std::round((VarParamWithLoc * 100.0) / VarParamTotal)
                   << "%\n";
      llvm::dbgs() << "PC Ranges covered: "
                   << (int)std::round((GlobalStats.ScopeBytesCovered * 100.0) /
                                      GlobalStats.ScopeBytes)
                   << "%\n");
  return true;
}
