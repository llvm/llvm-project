//===- RemarkDiff.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic tool to diff remarks based on properties
//
//===----------------------------------------------------------------------===//

#include "RemarkUtilHelpers.h"
#include "RemarkUtilRegistry.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ScopedPrinter.h"

namespace llvm {
namespace remarks {

enum ReportStyleOptions { human_output, json_output };
/// copy of Argument class using std::string instead of StringRef.
struct RemarkArgInfo {
  std::string Key;
  std::string Val;
  RemarkArgInfo(StringRef Key, StringRef Val)
      : Key(Key.str()), Val(Val.str()) {}
  void print(raw_ostream &OS) const;
};

hash_code hash_value(const RemarkArgInfo &Arg) {
  return hash_combine(Arg.Key, Arg.Val);
}

/// A wrapper for Remark class that can be used for generating remark diff.
struct RemarkInfo {
  std::string RemarkName;
  std::string FunctionName;
  std::string PassName;
  Type RemarkType;
  SmallVector<RemarkArgInfo, 4> Args;
  RemarkInfo();
  RemarkInfo(std::string RemarkName, std::string FunctionName,
             std::string PassName, Type RemarkType,
             SmallVector<RemarkArgInfo, 4> &Args)
      : RemarkName(RemarkName), FunctionName(FunctionName), PassName(PassName),
        RemarkType(RemarkType), Args(Args) {}
  RemarkInfo(Remark &Remark)
      : RemarkName(Remark.RemarkName.str()),
        FunctionName(Remark.FunctionName.str()),
        PassName(Remark.PassName.str()), RemarkType(Remark.RemarkType) {
    for (const auto &Arg : Remark.Args)
      Args.push_back({Arg.Key.str(), Arg.Val.str()});
  }

  /// Check if the remark has the same name, function name and pass name as \p
  /// RHS
  bool hasSameHeader(const RemarkInfo &RHS) const {
    return RemarkName == RHS.RemarkName && FunctionName == RHS.FunctionName &&
           PassName == RHS.PassName;
  };
  void print(raw_ostream &OS) const;
  void printHeader(raw_ostream &OS) const;
};

inline bool operator<(const RemarkArgInfo &LHS, const RemarkArgInfo &RHS) {
  return std::make_tuple(LHS.Key, LHS.Val) < std::make_tuple(RHS.Key, RHS.Val);
}

inline bool operator<(const RemarkInfo &LHS, const RemarkInfo &RHS) {
  return std::make_tuple(LHS.RemarkType, LHS.PassName, LHS.RemarkName,
                         LHS.FunctionName, LHS.Args) <
         std::make_tuple(RHS.RemarkType, RHS.PassName, RHS.RemarkName,
                         RHS.FunctionName, RHS.Args);
}

inline bool operator==(const RemarkArgInfo &LHS, const RemarkArgInfo &RHS) {
  return LHS.Key == RHS.Key && LHS.Val == RHS.Val;
}

inline bool operator==(const RemarkInfo &LHS, const RemarkInfo &RHS) {
  return LHS.RemarkName == RHS.RemarkName &&
         LHS.FunctionName == RHS.FunctionName && LHS.PassName == RHS.PassName &&
         LHS.RemarkType == RHS.RemarkType && LHS.Args == RHS.Args;
}

inline raw_ostream &operator<<(raw_ostream &OS,
                               const RemarkArgInfo &RemarkArgInfo) {
  RemarkArgInfo.print(OS);
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS, const RemarkInfo &RemarkInfo) {
  RemarkInfo.print(OS);
  return OS;
}

/// Represents the unique location where the remark was issued which is based on
/// the debug information attatched to the remark. The debug location conists of
/// the source file path, function name, line number and column number.
struct DebugLocation {
  std::string SourceFilePath;
  std::string FunctionName;
  unsigned SourceLine = 0;
  unsigned SourceColumn = 0;
  DebugLocation() = default;
  DebugLocation(StringRef SourceFilePath, StringRef FunctionName,
                unsigned SourceLine, unsigned SourceColumn)
      : SourceFilePath(SourceFilePath.str()), FunctionName(FunctionName.str()),
        SourceLine(SourceLine), SourceColumn(SourceColumn) {}
  std::string toString() {
    return "Ln: " + to_string(SourceLine) + " Col: " + to_string(SourceColumn);
  }
};

/// Configure the verbosity of the diff by choosing to only show unique remarks
/// from each version or only consider remarks if they differ in type or
/// argument. The configurator handles user specified arguments passed by flags.
struct DiffConfigurator {
  bool AddRemarksFromA;
  bool AddRemarksFromB;
  bool ShowCommonRemarks;
  bool ShowRemarkTypeDiff;
  bool ShowArgTypeDiff;
  DiffConfigurator(bool ShowArgDiffOnly, bool OnlyShowCommonRemarks,
                   bool OnlyShowDifferentRemarks, bool ShowOnlyA,
                   bool ShowOnlyB, bool ShowRemarkTypeDiffOnly) {
    AddRemarksFromA = !OnlyShowCommonRemarks && (ShowOnlyA || !ShowOnlyB);
    AddRemarksFromB = !OnlyShowCommonRemarks && (ShowOnlyB || !ShowOnlyA);
    ShowCommonRemarks = !OnlyShowDifferentRemarks || OnlyShowCommonRemarks;
    ShowRemarkTypeDiff = !ShowArgDiffOnly || ShowRemarkTypeDiffOnly;
    ShowArgTypeDiff = !ShowRemarkTypeDiffOnly || ShowArgDiffOnly;
  }
};

/// Represent a diff where the remark header information is the same but the
/// differ in remark type or agruments.
struct DiffAtRemark {
  RemarkInfo BaseRemark;
  std::optional<std::pair<Type, Type>> RemarkTypeDiff;
  SmallVector<RemarkArgInfo, 4> OnlyA;
  SmallVector<RemarkArgInfo, 4> OnlyB;
  SmallVector<RemarkArgInfo, 4> InBoth;

  /// Compute the diff between two remarks \p RA and \p RB which share the same
  /// header and differ in remark type and arguments.
  DiffAtRemark(RemarkInfo &RA, RemarkInfo &RB, DiffConfigurator &DiffConfig)
      : BaseRemark(RA) {
    if (DiffConfig.ShowArgTypeDiff) {
      unsigned ArgIdx = 0;
      // Loop through the remarks in RA and RB in order comparing both.
      for (; ArgIdx < std::min(RA.Args.size(), RB.Args.size()); ArgIdx++) {
        if (RA.Args[ArgIdx] == (RB.Args[ArgIdx]))
          InBoth.push_back(RA.Args[ArgIdx]);
        else {
          OnlyA.push_back(RA.Args[ArgIdx]);
          OnlyB.push_back(RB.Args[ArgIdx]);
        }
      }

      // Add the remaining remarks if they exist to OnlyA or OnlyB.
      SmallVector<RemarkArgInfo, 4> RemainingArgs =
          RA.Args.size() > RB.Args.size() ? RA.Args : RB.Args;
      bool IsARemaining = RA.Args.size() > RB.Args.size() ? true : false;
      for (; ArgIdx < RemainingArgs.size(); ArgIdx++)
        if (IsARemaining)
          OnlyA.push_back(RemainingArgs[ArgIdx]);
        else
          OnlyB.push_back(RemainingArgs[ArgIdx]);
    }

    // Compare remark type between RA and RB.
    if (DiffConfig.ShowRemarkTypeDiff && RA.RemarkType != RB.RemarkType) {
      RemarkTypeDiff = {RA.RemarkType, RB.RemarkType};
    }
  }

  void print(raw_ostream &OS) const;

  /// represent remark diff as a json object where the header is the same as the
  /// baseline remark and diff json key represents the differences between the
  /// two versions of the remark.
  json::Object toJson();
};

/// Represents the diff at a debug location. This can be unique remarks that
/// exist in file a or file b or remarks that share the same header but differ
/// in remark type or arguments. Any common remarks at the location are
/// discarded.
struct DiffAtLoc {
  DebugLocation Loc;
  SmallVector<RemarkInfo, 4> OnlyA;
  SmallVector<RemarkInfo, 4> OnlyB;
  // list of remarks that are different but share the same header.
  SmallVector<DiffAtRemark, 4> HasTheSameHeader;

  DiffAtLoc(DebugLocation Loc) : Loc(Loc) {}

  /// Check if the debug location is empty where no unique remarks exist
  /// in A, B or remarks sharing the same header but differ in type or
  /// arguments.
  bool isEmpty() {
    return OnlyA.empty() && OnlyB.empty() && HasTheSameHeader.empty();
  }
  void print(raw_ostream &OS);

  /// Display diff as a json object.
  json::Object toJson();
};

/// Represnt the diff between the two files as a list of diffs at each debug
/// location found in both remark files. The diff is filtered by user-specified
/// filters for remark name, pass name, function name and remark type.
struct Diff {
  SmallVector<DiffAtLoc, 8> DiffAtLocs;
  Filters &Filter;
  DiffConfigurator DiffConfig;
  Diff(Filters &Filter, DiffConfigurator DiffConfig)
      : Filter(Filter), DiffConfig(DiffConfig) {}
  /// Taking all debug locations represented in both files in \p DebugLocs
  /// calculate the difference between the remarks existing in each location in
  /// \p DebugLoc2RemarkA and \p DebugLoc2RemarkB.
  void computeDiff(
      SetVector<DebugLocation> &DebugLocs,
      MapVector<DebugLocation, SmallVector<RemarkInfo, 4>> &DebugLoc2RemarkA,
      MapVector<DebugLocation, SmallVector<RemarkInfo, 4>> &DebugLoc2RemarkB);
  /// Compute the diff between the remarks at a shared debug location between
  /// file a and b.
  void computeDiffAtLoc(DebugLocation Loc, ArrayRef<RemarkInfo> RemarksA,
                        ArrayRef<RemarkInfo> RemarksB);

  Error printDiff(StringRef InputFileNameA, StringRef InputFileNameB);
};
} // namespace remarks

template <> struct DenseMapInfo<remarks::DebugLocation, void> {
  static inline remarks::DebugLocation getEmptyKey() {
    return remarks::DebugLocation();
  }

  static inline remarks::DebugLocation getTombstoneKey() {
    auto Loc = remarks::DebugLocation();
    Loc.SourceFilePath = StringRef(
        reinterpret_cast<const char *>(~static_cast<uintptr_t>(1)), 0);
    Loc.FunctionName = StringRef(
        reinterpret_cast<const char *>(~static_cast<uintptr_t>(1)), 0);
    Loc.SourceColumn = ~0U - 1;
    Loc.SourceLine = ~0U - 1;
    return Loc;
  }

  static unsigned getHashValue(const remarks::DebugLocation &Key) {
    return hash_combine(Key.SourceFilePath, Key.FunctionName, Key.SourceLine,
                        Key.SourceColumn);
  }

  static bool isEqual(const remarks::DebugLocation &LHS,
                      const remarks::DebugLocation &RHS) {
    return std::make_tuple(LHS.SourceFilePath, LHS.FunctionName, LHS.SourceLine,
                           LHS.SourceColumn) ==
           std::make_tuple(RHS.SourceFilePath, RHS.FunctionName, RHS.SourceLine,
                           RHS.SourceColumn);
  }
};

template <> struct DenseMapInfo<remarks::RemarkInfo, void> {
  static inline remarks::RemarkInfo getEmptyKey() {
    return remarks::RemarkInfo();
  }

  static inline remarks::RemarkInfo getTombstoneKey() {
    auto Info = remarks::RemarkInfo();
    Info.RemarkName =
        reinterpret_cast<const char *>(~static_cast<uintptr_t>(1));
    Info.FunctionName =
        reinterpret_cast<const char *>(~static_cast<uintptr_t>(1));
    Info.PassName = reinterpret_cast<const char *>(~static_cast<uintptr_t>(1));
    Info.RemarkType = remarks::Type::Unknown;
    return Info;
  }

  static unsigned getHashValue(const remarks::RemarkInfo &Key) {
    auto ArgCode = hash_combine_range(Key.Args.begin(), Key.Args.end());
    return hash_combine(Key.RemarkName, Key.FunctionName, Key.PassName,
                        remarks::typeToStr(Key.RemarkType), ArgCode);
  }

  static bool isEqual(const remarks::RemarkInfo &LHS,
                      const remarks::RemarkInfo &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm