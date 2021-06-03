//=-- SampleProf.cpp - Sample profiling format support --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions used in the reading and writing of
// sample profile data.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/PseudoProbe.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <system_error>

using namespace llvm;
using namespace sampleprof;

static cl::opt<uint64_t> ProfileSymbolListCutOff(
    "profile-symbol-list-cutoff", cl::Hidden, cl::init(-1), cl::ZeroOrMore,
    cl::desc("Cutoff value about how many symbols in profile symbol list "
             "will be used. This is very useful for performance debugging"));

namespace llvm {
namespace sampleprof {
SampleProfileFormat FunctionSamples::Format;
bool FunctionSamples::ProfileIsProbeBased = false;
bool FunctionSamples::ProfileIsCS = false;
bool FunctionSamples::UseMD5 = false;
bool FunctionSamples::HasUniqSuffix = true;
bool FunctionSamples::ProfileIsFS = false;
} // namespace sampleprof
} // namespace llvm

namespace {

// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class SampleProfErrorCategoryType : public std::error_category {
  const char *name() const noexcept override { return "llvm.sampleprof"; }

  std::string message(int IE) const override {
    sampleprof_error E = static_cast<sampleprof_error>(IE);
    switch (E) {
    case sampleprof_error::success:
      return "Success";
    case sampleprof_error::bad_magic:
      return "Invalid sample profile data (bad magic)";
    case sampleprof_error::unsupported_version:
      return "Unsupported sample profile format version";
    case sampleprof_error::too_large:
      return "Too much profile data";
    case sampleprof_error::truncated:
      return "Truncated profile data";
    case sampleprof_error::malformed:
      return "Malformed sample profile data";
    case sampleprof_error::unrecognized_format:
      return "Unrecognized sample profile encoding format";
    case sampleprof_error::unsupported_writing_format:
      return "Profile encoding format unsupported for writing operations";
    case sampleprof_error::truncated_name_table:
      return "Truncated function name table";
    case sampleprof_error::not_implemented:
      return "Unimplemented feature";
    case sampleprof_error::counter_overflow:
      return "Counter overflow";
    case sampleprof_error::ostream_seek_unsupported:
      return "Ostream does not support seek";
    case sampleprof_error::compress_failed:
      return "Compress failure";
    case sampleprof_error::uncompress_failed:
      return "Uncompress failure";
    case sampleprof_error::zlib_unavailable:
      return "Zlib is unavailable";
    case sampleprof_error::hash_mismatch:
      return "Function hash mismatch";
    }
    llvm_unreachable("A value of sampleprof_error has no message.");
  }
};

} // end anonymous namespace

static ManagedStatic<SampleProfErrorCategoryType> ErrorCategory;

const std::error_category &llvm::sampleprof_category() {
  return *ErrorCategory;
}

void LineLocation::print(raw_ostream &OS) const {
  OS << LineOffset;
  if (Discriminator > 0)
    OS << "." << Discriminator;
}

raw_ostream &llvm::sampleprof::operator<<(raw_ostream &OS,
                                          const LineLocation &Loc) {
  Loc.print(OS);
  return OS;
}

/// Merge the samples in \p Other into this record.
/// Optionally scale sample counts by \p Weight.
sampleprof_error SampleRecord::merge(const SampleRecord &Other,
                                     uint64_t Weight) {
  sampleprof_error Result;
  // With pseudo probes, merge a dangling sample with a non-dangling sample
  // should result in a dangling sample.
  if (FunctionSamples::ProfileIsProbeBased &&
      (getSamples() == FunctionSamples::InvalidProbeCount ||
       Other.getSamples() == FunctionSamples::InvalidProbeCount)) {
    NumSamples = FunctionSamples::InvalidProbeCount;
    Result = sampleprof_error::success;
  } else {
    Result = addSamples(Other.getSamples(), Weight);
  }
  for (const auto &I : Other.getCallTargets()) {
    MergeResult(Result, addCalledTarget(I.first(), I.second, Weight));
  }
  return Result;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void LineLocation::dump() const { print(dbgs()); }
#endif

/// Print the sample record to the stream \p OS indented by \p Indent.
void SampleRecord::print(raw_ostream &OS, unsigned Indent) const {
  OS << NumSamples;
  if (hasCalls()) {
    OS << ", calls:";
    for (const auto &I : getSortedCallTargets())
      OS << " " << I.first << ":" << I.second;
  }
  OS << "\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SampleRecord::dump() const { print(dbgs(), 0); }
#endif

raw_ostream &llvm::sampleprof::operator<<(raw_ostream &OS,
                                          const SampleRecord &Sample) {
  Sample.print(OS, 0);
  return OS;
}

/// Print the samples collected for a function on stream \p OS.
void FunctionSamples::print(raw_ostream &OS, unsigned Indent) const {
  if (getFunctionHash())
    OS << "CFG checksum " << getFunctionHash() << "\n";

  OS << TotalSamples << ", " << TotalHeadSamples << ", " << BodySamples.size()
     << " sampled lines\n";

  OS.indent(Indent);
  if (!BodySamples.empty()) {
    OS << "Samples collected in the function's body {\n";
    SampleSorter<LineLocation, SampleRecord> SortedBodySamples(BodySamples);
    for (const auto &SI : SortedBodySamples.get()) {
      OS.indent(Indent + 2);
      OS << SI->first << ": " << SI->second;
    }
    OS.indent(Indent);
    OS << "}\n";
  } else {
    OS << "No samples collected in the function's body\n";
  }

  OS.indent(Indent);
  if (!CallsiteSamples.empty()) {
    OS << "Samples collected in inlined callsites {\n";
    SampleSorter<LineLocation, FunctionSamplesMap> SortedCallsiteSamples(
        CallsiteSamples);
    for (const auto &CS : SortedCallsiteSamples.get()) {
      for (const auto &FS : CS->second) {
        OS.indent(Indent + 2);
        OS << CS->first << ": inlined callee: " << FS.second.getName() << ": ";
        FS.second.print(OS, Indent + 4);
      }
    }
    OS.indent(Indent);
    OS << "}\n";
  } else {
    OS << "No inlined callsites in this function\n";
  }
}

raw_ostream &llvm::sampleprof::operator<<(raw_ostream &OS,
                                          const FunctionSamples &FS) {
  FS.print(OS);
  return OS;
}

unsigned FunctionSamples::getOffset(const DILocation *DIL) {
  return (DIL->getLine() - DIL->getScope()->getSubprogram()->getLine()) &
      0xffff;
}

LineLocation FunctionSamples::getCallSiteIdentifier(const DILocation *DIL) {
  if (FunctionSamples::ProfileIsProbeBased)
    // In a pseudo-probe based profile, a callsite is simply represented by the
    // ID of the probe associated with the call instruction. The probe ID is
    // encoded in the Discriminator field of the call instruction's debug
    // metadata.
    return LineLocation(PseudoProbeDwarfDiscriminator::extractProbeIndex(
                            DIL->getDiscriminator()),
                        0);
  else
    return LineLocation(FunctionSamples::getOffset(DIL),
                        DIL->getBaseDiscriminator());
}

const FunctionSamples *FunctionSamples::findFunctionSamples(
    const DILocation *DIL, SampleProfileReaderItaniumRemapper *Remapper) const {
  assert(DIL);
  SmallVector<std::pair<LineLocation, StringRef>, 10> S;

  const DILocation *PrevDIL = DIL;
  for (DIL = DIL->getInlinedAt(); DIL; DIL = DIL->getInlinedAt()) {
    unsigned Discriminator;
    if (ProfileIsFS)
      Discriminator = DIL->getDiscriminator();
    else
      Discriminator = DIL->getBaseDiscriminator();

    S.push_back(
        std::make_pair(LineLocation(getOffset(DIL), Discriminator),
                       PrevDIL->getScope()->getSubprogram()->getLinkageName()));
    PrevDIL = DIL;
  }
  if (S.size() == 0)
    return this;
  const FunctionSamples *FS = this;
  for (int i = S.size() - 1; i >= 0 && FS != nullptr; i--) {
    FS = FS->findFunctionSamplesAt(S[i].first, S[i].second, Remapper);
  }
  return FS;
}

void FunctionSamples::findAllNames(DenseSet<StringRef> &NameSet) const {
  NameSet.insert(Name);
  for (const auto &BS : BodySamples)
    for (const auto &TS : BS.second.getCallTargets())
      NameSet.insert(TS.getKey());

  for (const auto &CS : CallsiteSamples) {
    for (const auto &NameFS : CS.second) {
      NameSet.insert(NameFS.first);
      NameFS.second.findAllNames(NameSet);
    }
  }
}

const FunctionSamples *FunctionSamples::findFunctionSamplesAt(
    const LineLocation &Loc, StringRef CalleeName,
    SampleProfileReaderItaniumRemapper *Remapper) const {
  CalleeName = getCanonicalFnName(CalleeName);

  std::string CalleeGUID;
  CalleeName = getRepInFormat(CalleeName, UseMD5, CalleeGUID);

  auto iter = CallsiteSamples.find(Loc);
  if (iter == CallsiteSamples.end())
    return nullptr;
  auto FS = iter->second.find(CalleeName);
  if (FS != iter->second.end())
    return &FS->second;
  if (Remapper) {
    if (auto NameInProfile = Remapper->lookUpNameInProfile(CalleeName)) {
      auto FS = iter->second.find(*NameInProfile);
      if (FS != iter->second.end())
        return &FS->second;
    }
  }
  // If we cannot find exact match of the callee name, return the FS with
  // the max total count. Only do this when CalleeName is not provided,
  // i.e., only for indirect calls.
  if (!CalleeName.empty())
    return nullptr;
  uint64_t MaxTotalSamples = 0;
  const FunctionSamples *R = nullptr;
  for (const auto &NameFS : iter->second)
    if (NameFS.second.getTotalSamples() >= MaxTotalSamples) {
      MaxTotalSamples = NameFS.second.getTotalSamples();
      R = &NameFS.second;
    }
  return R;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void FunctionSamples::dump() const { print(dbgs(), 0); }
#endif

std::error_code ProfileSymbolList::read(const uint8_t *Data,
                                        uint64_t ListSize) {
  const char *ListStart = reinterpret_cast<const char *>(Data);
  uint64_t Size = 0;
  uint64_t StrNum = 0;
  while (Size < ListSize && StrNum < ProfileSymbolListCutOff) {
    StringRef Str(ListStart + Size);
    add(Str);
    Size += Str.size() + 1;
    StrNum++;
  }
  if (Size != ListSize && StrNum != ProfileSymbolListCutOff)
    return sampleprof_error::malformed;
  return sampleprof_error::success;
}

void SampleContextTrimmer::trimAndMergeColdContextProfiles(
    uint64_t ColdCountThreshold, bool TrimColdContext, bool MergeColdContext) {
  if (!TrimColdContext && !MergeColdContext)
    return;

  // Nothing to merge if sample threshold is zero
  if (ColdCountThreshold == 0)
    return;

  // Filter the cold profiles from ProfileMap and move them into a tmp
  // container
  std::vector<std::pair<StringRef, const FunctionSamples *>> ColdProfiles;
  for (const auto &I : ProfileMap) {
    const FunctionSamples &FunctionProfile = I.second;
    if (FunctionProfile.getTotalSamples() >= ColdCountThreshold)
      continue;
    ColdProfiles.emplace_back(I.getKey(), &I.second);
  }

  // Remove the cold profile from ProfileMap and merge them into BaseProileMap
  StringMap<FunctionSamples> BaseProfileMap;
  for (const auto &I : ColdProfiles) {
    if (MergeColdContext) {
      auto Ret = BaseProfileMap.try_emplace(
          I.second->getContext().getNameWithoutContext(), FunctionSamples());
      FunctionSamples &BaseProfile = Ret.first->second;
      BaseProfile.merge(*I.second);
    }
    ProfileMap.erase(I.first);
  }

  // Merge the base profiles into ProfileMap;
  for (const auto &I : BaseProfileMap) {
    // Filter the cold base profile
    if (TrimColdContext && I.second.getTotalSamples() < ColdCountThreshold &&
        ProfileMap.find(I.getKey()) == ProfileMap.end())
      continue;
    // Merge the profile if the original profile exists, otherwise just insert
    // as a new profile
    auto Ret = ProfileMap.try_emplace(I.getKey(), FunctionSamples());
    if (Ret.second) {
      SampleContext FContext(Ret.first->first(), RawContext);
      FunctionSamples &FProfile = Ret.first->second;
      FProfile.setContext(FContext);
      FProfile.setName(FContext.getNameWithoutContext());
    }
    FunctionSamples &OrigProfile = Ret.first->second;
    OrigProfile.merge(I.second);
  }
}

void SampleContextTrimmer::canonicalizeContextProfiles() {
  StringSet<> ProfilesToBeRemoved;
  // Note that StringMap order is guaranteed to be top-down order,
  // this makes sure we make room for promoted/merged context in the
  // map, before we move profiles in the map.
  for (auto &I : ProfileMap) {
    FunctionSamples &FProfile = I.second;
    StringRef ContextStr = FProfile.getNameWithContext();
    if (I.first() == ContextStr)
      continue;

    // Use the context string from FunctionSamples to update the keys of
    // ProfileMap. They can get out of sync after context profile promotion
    // through pre-inliner.
    auto Ret = ProfileMap.try_emplace(ContextStr, FProfile);
    assert(Ret.second && "Conext conflict during canonicalization");
    FProfile = Ret.first->second;

    // Track the context profile to remove
    ProfilesToBeRemoved.erase(ContextStr);
    ProfilesToBeRemoved.insert(I.first());
  }

  for (auto &I : ProfilesToBeRemoved) {
    ProfileMap.erase(I.first());
  }
}

std::error_code ProfileSymbolList::write(raw_ostream &OS) {
  // Sort the symbols before output. If doing compression.
  // It will make the compression much more effective.
  std::vector<StringRef> SortedList(Syms.begin(), Syms.end());
  llvm::sort(SortedList);

  std::string OutputString;
  for (auto &Sym : SortedList) {
    OutputString.append(Sym.str());
    OutputString.append(1, '\0');
  }

  OS << OutputString;
  return sampleprof_error::success;
}

void ProfileSymbolList::dump(raw_ostream &OS) const {
  OS << "======== Dump profile symbol list ========\n";
  std::vector<StringRef> SortedList(Syms.begin(), Syms.end());
  llvm::sort(SortedList);

  for (auto &Sym : SortedList)
    OS << Sym << "\n";
}
