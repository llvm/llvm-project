#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/Object/ObjectFile.h"

#define DEBUG_TYPE "dwarfdump"
using namespace llvm;
using namespace object;

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
  /// Number of variables wtih type.
  unsigned NumVarTypes = 0;
  /// Number of variables wtih DW_AT_location.
  unsigned NumVarLocations = 0;
};

/// Holds accumulated global statistics about DIEs.
struct GlobalStats {
  /// Total number of PC range bytes covered by DW_AT_locations.
  unsigned ScopeBytesCovered = 0;
  /// Total number of PC range bytes in each variable's enclosing scope,
  /// starting from the first definition of the variable.
  unsigned ScopeBytesFromFirstDefinition = 0;
  /// Total number of call site entries (DW_TAG_call_site) or
  /// (DW_AT_call_file & DW_AT_call_line).
  unsigned CallSiteEntries = 0;
  /// Total byte size of concrete functions. This byte size includes
  /// inline functions contained in the concrete functions.
  uint64_t FunctionSize = 0;
  /// Total byte size of inlined functions. This is the total number of bytes
  /// for the top inline functions within concrete functions. This can help
  /// tune the inline settings when compiling to match user expectations.
  uint64_t InlineFunctionSize = 0;
};

/// Extract the low pc from a Die.
static uint64_t getLowPC(DWARFDie Die) {
  auto RangesOrError = Die.getAddressRanges();
  DWARFAddressRangesVector Ranges;
  if (RangesOrError)
    Ranges = RangesOrError.get();
  else
    llvm::consumeError(RangesOrError.takeError());
  if (Ranges.size())
    return Ranges[0].LowPC;
  return dwarf::toAddress(Die.find(dwarf::DW_AT_low_pc), 0);
}

/// Collect debug info quality metrics for one DIE.
static void collectStatsForDie(DWARFDie Die, std::string FnPrefix,
                               std::string VarPrefix, uint64_t ScopeLowPC,
                               uint64_t BytesInScope, uint32_t InlineDepth,
                               StringMap<PerFunctionStats> &FnStatMap,
                               GlobalStats &GlobalStats) {
  bool HasLoc = false;
  bool HasSrcLoc = false;
  bool HasType = false;
  bool IsArtificial = false;
  uint64_t BytesCovered = 0;
  uint64_t OffsetToFirstDefinition = 0;

  if (Die.getTag() == dwarf::DW_TAG_call_site) {
    GlobalStats.CallSiteEntries++;
    return;
  }

  if (Die.getTag() != dwarf::DW_TAG_formal_parameter &&
      Die.getTag() != dwarf::DW_TAG_variable &&
      Die.getTag() != dwarf::DW_TAG_member) {
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
    auto FormValue = Die.find(dwarf::DW_AT_location);
    HasLoc = FormValue.hasValue();
    if (HasLoc) {
      // Get PC coverage.
      if (auto DebugLocOffset = FormValue->getAsSectionOffset()) {
        auto *DebugLoc = Die.getDwarfUnit()->getContext().getDebugLoc();
        if (auto List = DebugLoc->getLocationListAtOffset(*DebugLocOffset)) {
          for (auto Entry : List->Entries)
            BytesCovered += Entry.End - Entry.Begin;
          if (List->Entries.size()) {
            uint64_t FirstDef = List->Entries[0].Begin;
            uint64_t UnitOfs = getLowPC(Die.getDwarfUnit()->getUnitDIE());
            // Ranges sometimes start before the lexical scope.
            if (UnitOfs + FirstDef >= ScopeLowPC)
              OffsetToFirstDefinition = UnitOfs + FirstDef - ScopeLowPC;
            // Or even after it. Count that as a failure.
            if (OffsetToFirstDefinition > BytesInScope)
              OffsetToFirstDefinition = 0;
          }
        }
        assert(BytesInScope);
      } else {
        // Assume the entire range is covered by a single location.
        BytesCovered = BytesInScope;
      }
    }
  }

  // Collect PC range coverage data.
  auto &FnStats = FnStatMap[FnPrefix];
  if (DWARFDie D =
          Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_abstract_origin))
    Die = D;
  // By using the variable name + the path through the lexical block tree, the
  // keys are consistent across duplicate abstract origins in different CUs.
  std::string VarName = StringRef(Die.getName(DINameKind::ShortName));
  FnStats.VarsInFunction.insert(VarPrefix + VarName);
  if (BytesInScope) {
    FnStats.TotalVarWithLoc += (unsigned)HasLoc;
    // Adjust for the fact the variables often start their lifetime in the
    // middle of the scope.
    BytesInScope -= OffsetToFirstDefinition;
    // Turns out we have a lot of ranges that extend past the lexical scope.
    GlobalStats.ScopeBytesCovered += std::min(BytesInScope, BytesCovered);
    GlobalStats.ScopeBytesFromFirstDefinition += BytesInScope;
    assert(GlobalStats.ScopeBytesCovered <=
           GlobalStats.ScopeBytesFromFirstDefinition);
  } else if (Die.getTag() == dwarf::DW_TAG_member) {
    FnStats.ConstantMembers++;
  } else {
    FnStats.TotalVarWithLoc += (unsigned)HasLoc;
  }
  if (!IsArtificial) {
    if (Die.getTag() == dwarf::DW_TAG_formal_parameter) {
      FnStats.NumParams++;
      if (HasType)
        FnStats.NumParamTypes++;
      if (HasSrcLoc)
        FnStats.NumParamSourceLocations++;
      if (HasLoc)
        FnStats.NumParamLocations++;
    } else if (Die.getTag() == dwarf::DW_TAG_variable) {
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
                                  std::string VarPrefix, uint64_t ScopeLowPC,
                                  uint64_t BytesInScope, uint32_t InlineDepth,
                                  StringMap<PerFunctionStats> &FnStatMap,
                                  GlobalStats &GlobalStats) {
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
    ScopeLowPC = getLowPC(Die);

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
    collectStatsForDie(Die, FnPrefix, VarPrefix, ScopeLowPC, BytesInScope,
                       InlineDepth, FnStatMap, GlobalStats);
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

    collectStatsRecursive(Child, FnPrefix, ChildVarPrefix, ScopeLowPC,
                          BytesInScope, InlineDepth, FnStatMap, GlobalStats);
    Child = Child.getSibling();
  }
}

/// Print machine-readable output.
/// The machine-readable format is single-line JSON output.
/// \{
static void printDatum(raw_ostream &OS, const char *Key, StringRef Value) {
  OS << ",\"" << Key << "\":\"" << Value << '"';
  LLVM_DEBUG(llvm::dbgs() << Key << ": " << Value << '\n');
}
static void printDatum(raw_ostream &OS, const char *Key, uint64_t Value) {
  OS << ",\"" << Key << "\":" << Value;
  LLVM_DEBUG(llvm::dbgs() << Key << ": " << Value << '\n');
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
  StringMap<PerFunctionStats> Statistics;
  for (const auto &CU : static_cast<DWARFContext *>(&DICtx)->compile_units())
    if (DWARFDie CUDie = CU->getUnitDIE(false))
      collectStatsRecursive(CUDie, "/", "g", 0, 0, 0, Statistics, GlobalStats);

  /// The version number should be increased every time the algorithm is changed
  /// (including bug fixes). New metrics may be added without increasing the
  /// version.
  unsigned Version = 3;
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
  printDatum(OS, "scope bytes total",
             GlobalStats.ScopeBytesFromFirstDefinition);
  printDatum(OS, "scope bytes covered", GlobalStats.ScopeBytesCovered);
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
  OS << "}\n";
  LLVM_DEBUG(
      llvm::dbgs() << "Total Availability: "
                   << (int)std::round((VarParamWithLoc * 100.0) / VarParamTotal)
                   << "%\n";
      llvm::dbgs() << "PC Ranges covered: "
                   << (int)std::round((GlobalStats.ScopeBytesCovered * 100.0) /
                                      GlobalStats.ScopeBytesFromFirstDefinition)
                   << "%\n");
  return true;
}
