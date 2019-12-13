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
using namespace object;

/// This represents the number of categories of debug location coverage being
/// calculated. The first category is the number of variables with 0% location
/// coverage, but the last category is the number of variables with 100%
/// location coverage.
constexpr int NumOfCoverageCategories = 12;

/// Holds statistics for one function (or other entity that has a PC range and
/// contains variables, such as a compile unit).
struct PerFunctionStats {
  /// Number of inlined instances of this function.
  unsigned NumFnInlined = 0;
  /// Number of inlined instances that have abstract origins.
  unsigned NumAbstractOrigins = 0;
  /// Number of variables and parameters with location across all inlined
  /// instances.
  unsigned TotalVarWithLoc = 0;
  /// Number of constants with location across all inlined instances.
  unsigned ConstantMembers = 0;
  /// List of all Variables and parameters in this function.
  StringSet<> VarsInFunction;
  /// Compile units also cover a PC range, but have this flag set to false.
  bool IsFunction = false;
  /// Verify function definition has PC addresses (for detecting when
  /// a function has been inlined everywhere).
  bool HasPCAddresses = false;
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
  /// Number of variables.
  unsigned NumVars = 0;
  /// Number of variables with source location.
  unsigned NumVarSourceLocations = 0;
  /// Number of variables with type.
  unsigned NumVarTypes = 0;
  /// Number of variables with DW_AT_location.
  unsigned NumVarLocations = 0;
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
  /// Total number of PC range bytes in each variable's enclosing scope
  /// (only for parameters).
  unsigned ParamScopeBytes = 0;
  /// Total number of PC range bytes covered by DW_AT_locations with
  /// the debug entry values (DW_OP_entry_value) (only for parameters).
  unsigned ParamScopeEntryValueBytesCovered = 0;
  /// Total number of PC range bytes covered by DW_AT_locations (only for local
  /// variables).
  unsigned VarScopeBytesCovered = 0;
  /// Total number of PC range bytes in each variable's enclosing scope
  /// (only for local variables).
  unsigned VarScopeBytes = 0;
  /// Total number of PC range bytes covered by DW_AT_locations with
  /// the debug entry values (DW_OP_entry_value) (only for local variables).
  unsigned VarScopeEntryValueBytesCovered = 0;
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
  std::vector<unsigned> VarLocStats{
      std::vector<unsigned>(NumOfCoverageCategories, 0)};
  /// Map non debug entry values coverage for local variables.
  std::vector<unsigned> VarNonEntryValLocStats{
      std::vector<unsigned>(NumOfCoverageCategories, 0)};
  /// Total number of local variables and function parameters processed.
  unsigned NumVarParam = 0;
  /// Total number of formal parameters processed.
  unsigned NumParam = 0;
  /// Total number of local variables processed.
  unsigned NumVar = 0;
};

/// Collect debug location statistics for one DIE.
static void collectLocStats(uint64_t BytesCovered, uint64_t BytesInScope,
                            std::vector<unsigned> &VarParamLocStats,
                            std::vector<unsigned> &ParamLocStats,
                            std::vector<unsigned> &VarLocStats, bool IsParam,
                            bool IsLocalVar) {
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
    VarLocStats[CoverageBucket]++;
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
  bool IsArtificial = false;
  uint64_t BytesCovered = 0;
  uint64_t BytesEntryValuesCovered = 0;
  auto &FnStats = FnStatMap[FnPrefix];
  bool IsParam = Die.getTag() == dwarf::DW_TAG_formal_parameter;
  bool IsLocalVar = Die.getTag() == dwarf::DW_TAG_variable;

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

  if (!IsParam && !IsLocalVar && Die.getTag() != dwarf::DW_TAG_member) {
    // Not a variable or constant member.
    return;
  }

  if (Die.findRecursively(dwarf::DW_AT_decl_file) &&
      Die.findRecursively(dwarf::DW_AT_decl_line))
    HasSrcLoc = true;

  if (Die.findRecursively(dwarf::DW_AT_type))
    HasType = true;

  if (Die.find(dwarf::DW_AT_artificial))
    IsArtificial = true;

  auto IsEntryValue = [&](ArrayRef<uint8_t> D) -> bool {
    DWARFUnit *U = Die.getDwarfUnit();
    DataExtractor Data(toStringRef(D),
                       Die.getDwarfUnit()->getContext().isLittleEndian(), 0);
    DWARFExpression Expression(Data, U->getVersion(), U->getAddressByteSize());
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
    if (Die.getTag() == dwarf::DW_TAG_member) {
      // Non-const member.
      return;
    }
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
                    LocStats.ParamLocStats, LocStats.VarLocStats, IsParam,
                    IsLocalVar);
    // Non debug entry values coverage statistics.
    collectLocStats(BytesCovered - BytesEntryValuesCovered, BytesInScope,
                    LocStats.VarParamNonEntryValLocStats,
                    LocStats.ParamNonEntryValLocStats,
                    LocStats.VarNonEntryValLocStats, IsParam, IsLocalVar);
  }

  // Collect PC range coverage data.
  if (DWARFDie D =
          Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_abstract_origin))
    Die = D;
  // By using the variable name + the path through the lexical block tree, the
  // keys are consistent across duplicate abstract origins in different CUs.
  std::string VarName = StringRef(Die.getName(DINameKind::ShortName));
  FnStats.VarsInFunction.insert(VarPrefix + VarName);
  if (BytesInScope) {
    FnStats.TotalVarWithLoc += (unsigned)HasLoc;
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
      GlobalStats.VarScopeBytesCovered += std::min(BytesInScope, BytesCovered);
      GlobalStats.VarScopeBytes += BytesInScope;
      GlobalStats.VarScopeEntryValueBytesCovered += BytesEntryValuesCovered;
    }
    assert(GlobalStats.ScopeBytesCovered <= GlobalStats.ScopeBytes);
  } else if (Die.getTag() == dwarf::DW_TAG_member) {
    FnStats.ConstantMembers++;
  } else {
    FnStats.TotalVarWithLoc += (unsigned)HasLoc;
  }
  if (!IsArtificial) {
    if (IsParam) {
      FnStats.NumParams++;
      if (HasType)
        FnStats.NumParamTypes++;
      if (HasSrcLoc)
        FnStats.NumParamSourceLocations++;
      if (HasLoc)
        FnStats.NumParamLocations++;
    } else if (IsLocalVar) {
      FnStats.NumVars++;
      if (HasType)
        FnStats.NumVarTypes++;
      if (HasSrcLoc)
        FnStats.NumVarSourceLocations++;
      if (HasLoc)
        FnStats.NumVarLocations++;
    }
  }
}

/// Recursively collect debug info quality metrics.
static void collectStatsRecursive(DWARFDie Die, std::string FnPrefix,
                                  std::string VarPrefix, uint64_t BytesInScope,
                                  uint32_t InlineDepth,
                                  StringMap<PerFunctionStats> &FnStatMap,
                                  GlobalStats &GlobalStats,
                                  LocationStats &LocStats) {
  // Handle any kind of lexical scope.
  const dwarf::Tag Tag = Die.getTag();
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
      StringRef Name = Die.getName(DINameKind::LinkageName);
      if (Name.empty())
        Name = Die.getName(DINameKind::ShortName);
      FnPrefix = Name;
      // Skip over abstract origins.
      if (Die.find(dwarf::DW_AT_inline))
        return;
      // We've seen an (inlined) instance of this function.
      auto &FnStats = FnStatMap[Name];
      if (IsInlinedFunction) {
        FnStats.NumFnInlined++;
        if (Die.findRecursively(dwarf::DW_AT_abstract_origin))
          FnStats.NumAbstractOrigins++;
      }
      FnStats.IsFunction = true;
      if (BytesInThisScope && !IsInlinedFunction)
        FnStats.HasPCAddresses = true;
      std::string FnName = StringRef(Die.getName(DINameKind::ShortName));
      if (Die.findRecursively(dwarf::DW_AT_decl_file) &&
          Die.findRecursively(dwarf::DW_AT_decl_line))
        FnStats.HasSourceLocation = true;
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
  DWARFDie Child = Die.getFirstChild();
  while (Child) {
    std::string ChildVarPrefix = VarPrefix;
    if (Child.getTag() == dwarf::DW_TAG_lexical_block)
      ChildVarPrefix += toHex(LexicalBlockIndex++) + '.';

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
static void printLocationStats(raw_ostream &OS,
                               const char *Key,
                               std::vector<unsigned> &LocationStats) {
  OS << ",\"" << Key << " with 0% of its scope covered\":"
     << LocationStats[0];
  LLVM_DEBUG(llvm::dbgs() << Key << " with 0% of its scope covered: "
                          << LocationStats[0] << '\n');
  OS << ",\"" << Key << " with (0%,10%) of its scope covered\":"
     << LocationStats[1];
  LLVM_DEBUG(llvm::dbgs() << Key << " with (0%,10%) of its scope covered: "
                          << LocationStats[1] << '\n');
  for (unsigned i = 2; i < NumOfCoverageCategories - 1; ++i) {
    OS << ",\"" << Key << " with [" << (i - 1) * 10 << "%," << i * 10
       << "%) of its scope covered\":" << LocationStats[i];
    LLVM_DEBUG(llvm::dbgs()
               << Key << " with [" << (i - 1) * 10 << "%," << i * 10
               << "%) of its scope covered: " << LocationStats[i]);
  }
  OS << ",\"" << Key << " with 100% of its scope covered\":"
     << LocationStats[NumOfCoverageCategories - 1];
  LLVM_DEBUG(llvm::dbgs() << Key << " with 100% of its scope covered: "
                          << LocationStats[NumOfCoverageCategories - 1]);
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
bool collectStatsForObjectFile(ObjectFile &Obj, DWARFContext &DICtx,
                               Twine Filename, raw_ostream &OS) {
  StringRef FormatName = Obj.getFileFormatName();
  GlobalStats GlobalStats;
  LocationStats LocStats;
  StringMap<PerFunctionStats> Statistics;
  for (const auto &CU : static_cast<DWARFContext *>(&DICtx)->compile_units())
    if (DWARFDie CUDie = CU->getNonSkeletonUnitDIE(false))
      collectStatsRecursive(CUDie, "/", "g", 0, 0, Statistics, GlobalStats,
                            LocStats);

  /// The version number should be increased every time the algorithm is changed
  /// (including bug fixes). New metrics may be added without increasing the
  /// version.
  unsigned Version = 4;
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
  unsigned VarTotal = 0;
  unsigned VarWithType = 0;
  unsigned VarWithSrcLoc = 0;
  unsigned VarWithLoc = 0;
  for (auto &Entry : Statistics) {
    PerFunctionStats &Stats = Entry.getValue();
    unsigned TotalVars = Stats.VarsInFunction.size() * Stats.NumFnInlined;
    // Count variables in concrete out-of-line functions and in global scope.
    if (Stats.HasPCAddresses || !Stats.IsFunction)
      TotalVars += Stats.VarsInFunction.size();
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
    VarTotal += Stats.NumVars;
    VarWithType += Stats.NumVarTypes;
    VarWithLoc += Stats.NumVarLocations;
    VarWithSrcLoc += Stats.NumVarSourceLocations;
  }

  // Print summary.
  OS.SetBufferSize(1024);
  OS << "{\"version\":" << Version;
  LLVM_DEBUG(llvm::dbgs() << "Variable location quality metrics\n";
             llvm::dbgs() << "---------------------------------\n");
  printDatum(OS, "file", Filename.str());
  printDatum(OS, "format", FormatName);
  printDatum(OS, "source functions", NumFunctions);
  printDatum(OS, "source functions with location", NumFuncsWithSrcLoc);
  printDatum(OS, "inlined functions", NumInlinedFunctions);
  printDatum(OS, "inlined funcs with abstract origins", NumAbstractOrigins);
  printDatum(OS, "unique source variables", VarParamUnique);
  printDatum(OS, "source variables", VarParamTotal);
  printDatum(OS, "variables with location", VarParamWithLoc);
  printDatum(OS, "call site entries", GlobalStats.CallSiteEntries);
  printDatum(OS, "call site DIEs", GlobalStats.CallSiteDIEs);
  printDatum(OS, "call site parameter DIEs", GlobalStats.CallSiteParamDIEs);
  printDatum(OS, "scope bytes total", GlobalStats.ScopeBytes);
  printDatum(OS, "scope bytes covered", GlobalStats.ScopeBytesCovered);
  printDatum(OS, "entry value scope bytes covered",
             GlobalStats.ScopeEntryValueBytesCovered);
  printDatum(OS, "formal params scope bytes total",
             GlobalStats.ParamScopeBytes);
  printDatum(OS, "formal params scope bytes covered",
             GlobalStats.ParamScopeBytesCovered);
  printDatum(OS, "formal params entry value scope bytes covered",
             GlobalStats.ParamScopeEntryValueBytesCovered);
  printDatum(OS, "vars scope bytes total", GlobalStats.VarScopeBytes);
  printDatum(OS, "vars scope bytes covered", GlobalStats.VarScopeBytesCovered);
  printDatum(OS, "vars entry value scope bytes covered",
             GlobalStats.VarScopeEntryValueBytesCovered);
  printDatum(OS, "total function size", GlobalStats.FunctionSize);
  printDatum(OS, "total inlined function size", GlobalStats.InlineFunctionSize);
  printDatum(OS, "total formal params", ParamTotal);
  printDatum(OS, "formal params with source location", ParamWithSrcLoc);
  printDatum(OS, "formal params with type", ParamWithType);
  printDatum(OS, "formal params with binary location", ParamWithLoc);
  printDatum(OS, "total vars", VarTotal);
  printDatum(OS, "vars with source location", VarWithSrcLoc);
  printDatum(OS, "vars with type", VarWithType);
  printDatum(OS, "vars with binary location", VarWithLoc);
  printDatum(OS, "total variables procesed by location statistics",
             LocStats.NumVarParam);
  printLocationStats(OS, "variables", LocStats.VarParamLocStats);
  printLocationStats(OS, "variables (excluding the debug entry values)",
                     LocStats.VarParamNonEntryValLocStats);
  printDatum(OS, "total params procesed by location statistics",
             LocStats.NumParam);
  printLocationStats(OS, "params", LocStats.ParamLocStats);
  printLocationStats(OS, "params (excluding the debug entry values)",
                     LocStats.ParamNonEntryValLocStats);
  printDatum(OS, "total vars procesed by location statistics", LocStats.NumVar);
  printLocationStats(OS, "vars", LocStats.VarLocStats);
  printLocationStats(OS, "vars (excluding the debug entry values)",
                     LocStats.VarNonEntryValLocStats);
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
