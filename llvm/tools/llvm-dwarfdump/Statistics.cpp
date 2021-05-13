//===-- Statistics.cpp - Debug Info quality metrics -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-dwarfdump.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "dwarfdump"
using namespace llvm;
using namespace llvm::dwarfdump;
using namespace llvm::object;

namespace {
/// This represents the number of categories of debug location coverage being
/// calculated. The first category is the number of variables with 0% location
/// coverage, but the last category is the number of variables with 100%
/// location coverage.
constexpr int NumOfCoverageCategories = 12;

/// This is used for zero location coverage bucket.
constexpr unsigned ZeroCoverageBucket = 0;

/// This represents variables DIE offsets.
using AbstractOriginVarsTy = llvm::SmallVector<uint64_t>;
/// This maps function DIE offset to its variables.
using AbstractOriginVarsTyMap = llvm::DenseMap<uint64_t, AbstractOriginVarsTy>;
/// This represents function DIE offsets containing an abstract_origin.
using FunctionsWithAbstractOriginTy = llvm::SmallVector<uint64_t>;

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
  unsigned TotalBytesCovered = 0;
  /// Total number of parent DIE PC range bytes covered by DW_AT_Locations.
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
static void collectLocStats(uint64_t ScopeBytesCovered, uint64_t BytesInScope,
                            std::vector<unsigned> &VarParamLocStats,
                            std::vector<unsigned> &ParamLocStats,
                            std::vector<unsigned> &LocalVarLocStats,
                            bool IsParam, bool IsLocalVar) {
  auto getCoverageBucket = [ScopeBytesCovered, BytesInScope]() -> unsigned {
    // No debug location at all for the variable.
    if (ScopeBytesCovered == 0)
      return 0;
    // Fully covered variable within its scope.
    if (ScopeBytesCovered >= BytesInScope)
      return NumOfCoverageCategories - 1;
    // Get covered range (e.g. 20%-29%).
    unsigned LocBucket = 100 * (double)ScopeBytesCovered / BytesInScope;
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

/// Return the number of bytes in the overlap of ranges A and B.
static uint64_t calculateOverlap(DWARFAddressRange A, DWARFAddressRange B) {
  uint64_t Lower = std::max(A.LowPC, B.LowPC);
  uint64_t Upper = std::min(A.HighPC, B.HighPC);
  if (Lower >= Upper)
    return 0;
  return Upper - Lower;
}

/// Collect debug info quality metrics for one DIE.
static void collectStatsForDie(DWARFDie Die, const std::string &FnPrefix,
                               const std::string &VarPrefix,
                               uint64_t BytesInScope, uint32_t InlineDepth,
                               StringMap<PerFunctionStats> &FnStatMap,
                               GlobalStats &GlobalStats,
                               LocationStats &LocStats,
                               AbstractOriginVarsTy *AbstractOriginVariables) {
  const dwarf::Tag Tag = Die.getTag();
  // Skip CU node.
  if (Tag == dwarf::DW_TAG_compile_unit)
    return;

  bool HasLoc = false;
  bool HasSrcLoc = false;
  bool HasType = false;
  uint64_t TotalBytesCovered = 0;
  uint64_t ScopeBytesCovered = 0;
  uint64_t BytesEntryValuesCovered = 0;
  auto &FnStats = FnStatMap[FnPrefix];
  bool IsParam = Tag == dwarf::DW_TAG_formal_parameter;
  bool IsLocalVar = Tag == dwarf::DW_TAG_variable;
  bool IsConstantMember = Tag == dwarf::DW_TAG_member &&
                          Die.find(dwarf::DW_AT_const_value);

  // For zero covered inlined variables the locstats will be
  // calculated later.
  bool DeferLocStats = false;

  if (Tag == dwarf::DW_TAG_call_site || Tag == dwarf::DW_TAG_GNU_call_site) {
    GlobalStats.CallSiteDIEs++;
    return;
  }

  if (Tag == dwarf::DW_TAG_call_site_parameter ||
      Tag == dwarf::DW_TAG_GNU_call_site_parameter) {
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

  if (Die.find(dwarf::DW_AT_abstract_origin)) {
    if (Die.find(dwarf::DW_AT_location) || Die.find(dwarf::DW_AT_const_value)) {
      if (AbstractOriginVariables) {
        auto Offset = Die.find(dwarf::DW_AT_abstract_origin);
        // Do not track this variable any more, since it has location
        // coverage.
        llvm::erase_value(*AbstractOriginVariables, (*Offset).getRawUValue());
      }
    } else {
      // The locstats will be handled at the end of
      // the collectStatsRecursive().
      DeferLocStats = true;
    }
  }

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
    ScopeBytesCovered = BytesInScope;
    TotalBytesCovered = BytesInScope;
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
        ScopeBytesCovered = BytesInScope;
        TotalBytesCovered = BytesInScope;
      } else {
        // Caller checks this Expected result already, it cannot fail.
        auto ScopeRanges = cantFail(Die.getParent().getAddressRanges());
        for (auto Entry : *Loc) {
          TotalBytesCovered += Entry.Range->HighPC - Entry.Range->LowPC;
          uint64_t ScopeBytesCoveredByEntry = 0;
          // Calculate how many bytes of the parent scope this entry covers.
          // FIXME: In section 2.6.2 of the DWARFv5 spec it says that "The
          // address ranges defined by the bounded location descriptions of a
          // location list may overlap". So in theory a variable can have
          // multiple simultaneous locations, which would make this calculation
          // misleading because we will count the overlapped areas
          // twice. However, clang does not currently emit DWARF like this.
          for (DWARFAddressRange R : ScopeRanges) {
            ScopeBytesCoveredByEntry += calculateOverlap(*Entry.Range, R);
          }
          ScopeBytesCovered += ScopeBytesCoveredByEntry;
          if (IsEntryValue(Entry.Expr))
            BytesEntryValuesCovered += ScopeBytesCoveredByEntry;
        }
      }
    }
  }

  // Calculate the debug location statistics.
  if (BytesInScope && !DeferLocStats) {
    LocStats.NumVarParam++;
    if (IsParam)
      LocStats.NumParam++;
    else if (IsLocalVar)
      LocStats.NumVar++;

    collectLocStats(ScopeBytesCovered, BytesInScope, LocStats.VarParamLocStats,
                    LocStats.ParamLocStats, LocStats.LocalVarLocStats, IsParam,
                    IsLocalVar);
    // Non debug entry values coverage statistics.
    collectLocStats(ScopeBytesCovered - BytesEntryValuesCovered, BytesInScope,
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

  GlobalStats.TotalBytesCovered += TotalBytesCovered;
  if (BytesInScope) {
    GlobalStats.ScopeBytesCovered += ScopeBytesCovered;
    GlobalStats.ScopeBytes += BytesInScope;
    GlobalStats.ScopeEntryValueBytesCovered += BytesEntryValuesCovered;
    if (IsParam) {
      GlobalStats.ParamScopeBytesCovered += ScopeBytesCovered;
      GlobalStats.ParamScopeBytes += BytesInScope;
      GlobalStats.ParamScopeEntryValueBytesCovered += BytesEntryValuesCovered;
    } else if (IsLocalVar) {
      GlobalStats.LocalVarScopeBytesCovered += ScopeBytesCovered;
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

/// Recursively collect variables from subprogram with DW_AT_inline attribute.
static void collectAbstractOriginFnInfo(
    DWARFDie Die, uint64_t SPOffset,
    AbstractOriginVarsTyMap &GlobalAbstractOriginFnInfo) {
  DWARFDie Child = Die.getFirstChild();
  while (Child) {
    const dwarf::Tag ChildTag = Child.getTag();
    if (ChildTag == dwarf::DW_TAG_formal_parameter ||
        ChildTag == dwarf::DW_TAG_variable)
      GlobalAbstractOriginFnInfo[SPOffset].push_back(Child.getOffset());
    else if (ChildTag == dwarf::DW_TAG_lexical_block)
      collectAbstractOriginFnInfo(Child, SPOffset, GlobalAbstractOriginFnInfo);
    Child = Child.getSibling();
  }
}

/// Recursively collect debug info quality metrics.
static void collectStatsRecursive(
    DWARFDie Die, std::string FnPrefix, std::string VarPrefix,
    uint64_t BytesInScope, uint32_t InlineDepth,
    StringMap<PerFunctionStats> &FnStatMap, GlobalStats &GlobalStats,
    LocationStats &LocStats,
    AbstractOriginVarsTyMap &GlobalAbstractOriginFnInfo,
    FunctionsWithAbstractOriginTy &FnsWithAbstractOriginToBeProcessed,
    AbstractOriginVarsTy *AbstractOriginVarsPtr = nullptr) {
  // Skip NULL nodes.
  if (Die.isNULL())
    return;

  const dwarf::Tag Tag = Die.getTag();
  // Skip function types.
  if (Tag == dwarf::DW_TAG_subroutine_type)
    return;

  // Handle any kind of lexical scope.
  const bool HasAbstractOrigin = Die.find(dwarf::DW_AT_abstract_origin) != None;
  const bool IsFunction = Tag == dwarf::DW_TAG_subprogram;
  const bool IsBlock = Tag == dwarf::DW_TAG_lexical_block;
  const bool IsInlinedFunction = Tag == dwarf::DW_TAG_inlined_subroutine;
  // We want to know how many variables (with abstract_origin) don't have
  // location info.
  const bool IsCandidateForZeroLocCovTracking =
      (IsInlinedFunction || (IsFunction && HasAbstractOrigin));

  AbstractOriginVarsTy AbstractOriginVars;

  // Get the vars of the inlined fn, so the locstats
  // reports the missing vars (with coverage 0%).
  if (IsCandidateForZeroLocCovTracking) {
    auto OffsetFn = Die.find(dwarf::DW_AT_abstract_origin);
    if (OffsetFn) {
      uint64_t OffsetOfInlineFnCopy = (*OffsetFn).getRawUValue();
      if (GlobalAbstractOriginFnInfo.count(OffsetOfInlineFnCopy)) {
        AbstractOriginVars = GlobalAbstractOriginFnInfo[OffsetOfInlineFnCopy];
        AbstractOriginVarsPtr = &AbstractOriginVars;
      } else {
        // This means that the DW_AT_inline fn copy is out of order,
        // so this abstract origin instance will be processed later.
        FnsWithAbstractOriginToBeProcessed.push_back(Die.getOffset());
        AbstractOriginVarsPtr = nullptr;
      }
    }
  }

  if (IsFunction || IsInlinedFunction || IsBlock) {
    // Reset VarPrefix when entering a new function.
    if (IsFunction || IsInlinedFunction)
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
      // Skip over abstract origins, but collect variables
      // from it so it can be used for location statistics
      // for inlined instancies.
      if (Die.find(dwarf::DW_AT_inline)) {
        uint64_t SPOffset = Die.getOffset();
        collectAbstractOriginFnInfo(Die, SPOffset, GlobalAbstractOriginFnInfo);
        return;
      }

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
                       FnStatMap, GlobalStats, LocStats, AbstractOriginVarsPtr);
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

    collectStatsRecursive(
        Child, FnPrefix, ChildVarPrefix, BytesInScope, InlineDepth, FnStatMap,
        GlobalStats, LocStats, GlobalAbstractOriginFnInfo,
        FnsWithAbstractOriginToBeProcessed, AbstractOriginVarsPtr);
    Child = Child.getSibling();
  }

  if (!IsCandidateForZeroLocCovTracking)
    return;

  // After we have processed all vars of the inlined function (or function with
  // an abstract_origin), we want to know how many variables have no location.
  for (auto Offset : AbstractOriginVars) {
    LocStats.NumVarParam++;
    LocStats.VarParamLocStats[ZeroCoverageBucket]++;
    auto FnDie = Die.getDwarfUnit()->getDIEForOffset(Offset);
    if (!FnDie)
      continue;
    auto Tag = FnDie.getTag();
    if (Tag == dwarf::DW_TAG_formal_parameter) {
      LocStats.NumParam++;
      LocStats.ParamLocStats[ZeroCoverageBucket]++;
    } else if (Tag == dwarf::DW_TAG_variable) {
      LocStats.NumVar++;
      LocStats.LocalVarLocStats[ZeroCoverageBucket]++;
    }
  }
}

/// Print human-readable output.
/// \{
static void printDatum(json::OStream &J, const char *Key, json::Value Value) {
  J.attribute(Key, Value);
  LLVM_DEBUG(llvm::dbgs() << Key << ": " << Value << '\n');
}

static void printLocationStats(json::OStream &J, const char *Key,
                               std::vector<unsigned> &LocationStats) {
  J.attribute(
      (Twine(Key) + " with 0% of parent scope covered by DW_AT_location").str(),
      LocationStats[0]);
  LLVM_DEBUG(
      llvm::dbgs() << Key
                   << " with 0% of parent scope covered by DW_AT_location: \\"
                   << LocationStats[0] << '\n');
  J.attribute(
      (Twine(Key) + " with (0%,10%) of parent scope covered by DW_AT_location")
          .str(),
      LocationStats[1]);
  LLVM_DEBUG(llvm::dbgs()
             << Key
             << " with (0%,10%) of parent scope covered by DW_AT_location: "
             << LocationStats[1] << '\n');
  for (unsigned i = 2; i < NumOfCoverageCategories - 1; ++i) {
    J.attribute((Twine(Key) + " with [" + Twine((i - 1) * 10) + "%," +
                 Twine(i * 10) + "%) of parent scope covered by DW_AT_location")
                    .str(),
                LocationStats[i]);
    LLVM_DEBUG(llvm::dbgs()
               << Key << " with [" << (i - 1) * 10 << "%," << i * 10
               << "%) of parent scope covered by DW_AT_location: "
               << LocationStats[i]);
  }
  J.attribute(
      (Twine(Key) + " with 100% of parent scope covered by DW_AT_location")
          .str(),
      LocationStats[NumOfCoverageCategories - 1]);
  LLVM_DEBUG(
      llvm::dbgs() << Key
                   << " with 100% of parent scope covered by DW_AT_location: "
                   << LocationStats[NumOfCoverageCategories - 1]);
}

static void printSectionSizes(json::OStream &J, const SectionSizes &Sizes) {
  for (const auto &It : Sizes.DebugSectionSizes)
    J.attribute((Twine("#bytes in ") + It.first).str(), int64_t(It.second));
}

/// Stop tracking variables that contain abstract_origin with a location.
/// This is used for out-of-order DW_AT_inline subprograms only.
static void updateVarsWithAbstractOriginLocCovInfo(
    DWARFDie FnDieWithAbstractOrigin,
    AbstractOriginVarsTy &AbstractOriginVars) {
  DWARFDie Child = FnDieWithAbstractOrigin.getFirstChild();
  while (Child) {
    const dwarf::Tag ChildTag = Child.getTag();
    if ((ChildTag == dwarf::DW_TAG_formal_parameter ||
         ChildTag == dwarf::DW_TAG_variable) &&
        (Child.find(dwarf::DW_AT_location) ||
         Child.find(dwarf::DW_AT_const_value))) {
      auto OffsetVar = Child.find(dwarf::DW_AT_abstract_origin);
      if (OffsetVar)
        llvm::erase_value(AbstractOriginVars, (*OffsetVar).getRawUValue());
    } else if (ChildTag == dwarf::DW_TAG_lexical_block)
      updateVarsWithAbstractOriginLocCovInfo(Child, AbstractOriginVars);
    Child = Child.getSibling();
  }
}

/// Collect zero location coverage for inlined variables which refer to
/// a DW_AT_inline copy of subprogram that is out of order in the DWARF.
/// Also cover the variables of a concrete function (represented with
/// the DW_TAG_subprogram) with an abstract_origin attribute.
static void collectZeroLocCovForVarsWithAbstractOrigin(
    DWARFUnit *DwUnit, GlobalStats &GlobalStats, LocationStats &LocStats,
    AbstractOriginVarsTyMap &GlobalAbstractOriginFnInfo,
    FunctionsWithAbstractOriginTy &FnsWithAbstractOriginToBeProcessed) {
  for (auto FnOffset : FnsWithAbstractOriginToBeProcessed) {
    DWARFDie FnDieWithAbstractOrigin = DwUnit->getDIEForOffset(FnOffset);
    auto FnCopy = FnDieWithAbstractOrigin.find(dwarf::DW_AT_abstract_origin);
    AbstractOriginVarsTy AbstractOriginVars;
    if (!FnCopy)
      continue;

    AbstractOriginVars = GlobalAbstractOriginFnInfo[(*FnCopy).getRawUValue()];
    updateVarsWithAbstractOriginLocCovInfo(FnDieWithAbstractOrigin,
                                           AbstractOriginVars);

    for (auto Offset : AbstractOriginVars) {
      LocStats.NumVarParam++;
      LocStats.VarParamLocStats[ZeroCoverageBucket]++;
      auto Tag = DwUnit->getDIEForOffset(Offset).getTag();
      if (Tag == dwarf::DW_TAG_formal_parameter) {
        LocStats.NumParam++;
        LocStats.ParamLocStats[ZeroCoverageBucket]++;
      } else if (Tag == dwarf::DW_TAG_variable) {
        LocStats.NumVar++;
        LocStats.LocalVarLocStats[ZeroCoverageBucket]++;
      }
    }
  }
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
  for (const auto &CU : static_cast<DWARFContext *>(&DICtx)->compile_units()) {
    if (DWARFDie CUDie = CU->getNonSkeletonUnitDIE(false)) {
      // These variables are being reset for each CU, since there could be
      // a situation where we have two subprogram DIEs with the same offsets
      // in two diferent CUs, and we can end up using wrong variables info
      // when trying to resolve abstract_origin attribute.
      // TODO: Handle LTO cases where the abstract origin of
      // the function is in a different CU than the one it's
      // referenced from or inlined into.
      AbstractOriginVarsTyMap GlobalAbstractOriginFnInfo;
      FunctionsWithAbstractOriginTy FnsWithAbstractOriginToBeProcessed;

      collectStatsRecursive(CUDie, "/", "g", 0, 0, Statistics, GlobalStats,
                            LocStats, GlobalAbstractOriginFnInfo,
                            FnsWithAbstractOriginToBeProcessed);

      collectZeroLocCovForVarsWithAbstractOrigin(
          CUDie.getDwarfUnit(), GlobalStats, LocStats,
          GlobalAbstractOriginFnInfo, FnsWithAbstractOriginToBeProcessed);
    }
  }

  /// Collect the sizes of debug sections.
  SectionSizes Sizes;
  calculateSectionSizes(Obj, Sizes, Filename);

  /// The version number should be increased every time the algorithm is changed
  /// (including bug fixes). New metrics may be added without increasing the
  /// version.
  unsigned Version = 8;
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
  json::OStream J(OS, 2);
  J.objectBegin();
  J.attribute("version", Version);
  LLVM_DEBUG(llvm::dbgs() << "Variable location quality metrics\n";
             llvm::dbgs() << "---------------------------------\n");

  printDatum(J, "file", Filename.str());
  printDatum(J, "format", FormatName);

  printDatum(J, "#functions", NumFunctions);
  printDatum(J, "#functions with location", NumFuncsWithSrcLoc);
  printDatum(J, "#inlined functions", NumInlinedFunctions);
  printDatum(J, "#inlined functions with abstract origins", NumAbstractOrigins);

  // This includes local variables and formal parameters.
  printDatum(J, "#unique source variables", VarParamUnique);
  printDatum(J, "#source variables", VarParamTotal);
  printDatum(J, "#source variables with location", VarParamWithLoc);

  printDatum(J, "#call site entries", GlobalStats.CallSiteEntries);
  printDatum(J, "#call site DIEs", GlobalStats.CallSiteDIEs);
  printDatum(J, "#call site parameter DIEs", GlobalStats.CallSiteParamDIEs);

  printDatum(J, "sum_all_variables(#bytes in parent scope)",
             GlobalStats.ScopeBytes);
  printDatum(J,
             "sum_all_variables(#bytes in any scope covered by DW_AT_location)",
             GlobalStats.TotalBytesCovered);
  printDatum(J,
             "sum_all_variables(#bytes in parent scope covered by "
             "DW_AT_location)",
             GlobalStats.ScopeBytesCovered);
  printDatum(J,
             "sum_all_variables(#bytes in parent scope covered by "
             "DW_OP_entry_value)",
             GlobalStats.ScopeEntryValueBytesCovered);

  printDatum(J, "sum_all_params(#bytes in parent scope)",
             GlobalStats.ParamScopeBytes);
  printDatum(J,
             "sum_all_params(#bytes in parent scope covered by DW_AT_location)",
             GlobalStats.ParamScopeBytesCovered);
  printDatum(J,
             "sum_all_params(#bytes in parent scope covered by "
             "DW_OP_entry_value)",
             GlobalStats.ParamScopeEntryValueBytesCovered);

  printDatum(J, "sum_all_local_vars(#bytes in parent scope)",
             GlobalStats.LocalVarScopeBytes);
  printDatum(J,
             "sum_all_local_vars(#bytes in parent scope covered by "
             "DW_AT_location)",
             GlobalStats.LocalVarScopeBytesCovered);
  printDatum(J,
             "sum_all_local_vars(#bytes in parent scope covered by "
             "DW_OP_entry_value)",
             GlobalStats.LocalVarScopeEntryValueBytesCovered);

  printDatum(J, "#bytes within functions", GlobalStats.FunctionSize);
  printDatum(J, "#bytes within inlined functions",
             GlobalStats.InlineFunctionSize);

  // Print the summary for formal parameters.
  printDatum(J, "#params", ParamTotal);
  printDatum(J, "#params with source location", ParamWithSrcLoc);
  printDatum(J, "#params with type", ParamWithType);
  printDatum(J, "#params with binary location", ParamWithLoc);

  // Print the summary for local variables.
  printDatum(J, "#local vars", LocalVarTotal);
  printDatum(J, "#local vars with source location", LocalVarWithSrcLoc);
  printDatum(J, "#local vars with type", LocalVarWithType);
  printDatum(J, "#local vars with binary location", LocalVarWithLoc);

  // Print the debug section sizes.
  printSectionSizes(J, Sizes);

  // Print the location statistics for variables (includes local variables
  // and formal parameters).
  printDatum(J, "#variables processed by location statistics",
             LocStats.NumVarParam);
  printLocationStats(J, "#variables", LocStats.VarParamLocStats);
  printLocationStats(J, "#variables - entry values",
                     LocStats.VarParamNonEntryValLocStats);

  // Print the location statistics for formal parameters.
  printDatum(J, "#params processed by location statistics", LocStats.NumParam);
  printLocationStats(J, "#params", LocStats.ParamLocStats);
  printLocationStats(J, "#params - entry values",
                     LocStats.ParamNonEntryValLocStats);

  // Print the location statistics for local variables.
  printDatum(J, "#local vars processed by location statistics",
             LocStats.NumVar);
  printLocationStats(J, "#local vars", LocStats.LocalVarLocStats);
  printLocationStats(J, "#local vars - entry values",
                     LocStats.LocalVarNonEntryValLocStats);
  J.objectEnd();
  OS << '\n';
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
