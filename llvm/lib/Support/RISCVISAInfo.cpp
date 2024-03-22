//===-- RISCVISAInfo.cpp - RISC-V Arch String Parser ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <atomic>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;

namespace {

struct RISCVSupportedExtension {
  const char *Name;
  /// Supported version.
  RISCVISAInfo::ExtensionVersion Version;

  bool operator<(const RISCVSupportedExtension &RHS) const {
    return StringRef(Name) < StringRef(RHS.Name);
  }
};

} // end anonymous namespace

static constexpr StringLiteral AllStdExts = "mafdqlcbkjtpvnh";

static const char *RISCVGImplications[] = {
  "i", "m", "a", "f", "d", "zicsr", "zifencei"
};

// NOTE: This table should be sorted alphabetically by extension name.
static const RISCVSupportedExtension SupportedExtensions[] = {
    {"a", {2, 1}},
    {"c", {2, 0}},
    {"d", {2, 2}},
    {"e", {2, 0}},
    {"f", {2, 2}},
    {"h", {1, 0}},
    {"i", {2, 1}},
    {"m", {2, 0}},

    {"shcounterenw", {1, 0}},
    {"shgatpa", {1, 0}},
    {"shtvala", {1, 0}},
    {"shvsatpa", {1, 0}},
    {"shvstvala", {1, 0}},
    {"shvstvecd", {1, 0}},
    {"smaia", {1, 0}},
    {"smepmp", {1, 0}},
    {"ssaia", {1, 0}},
    {"ssccptr", {1, 0}},
    {"sscofpmf", {1, 0}},
    {"sscounterenw", {1, 0}},
    {"ssstateen", {1, 0}},
    {"ssstrict", {1, 0}},
    {"sstc", {1, 0}},
    {"sstvala", {1, 0}},
    {"sstvecd", {1, 0}},
    {"ssu64xl", {1, 0}},
    {"svade", {1, 0}},
    {"svadu", {1, 0}},
    {"svbare", {1, 0}},
    {"svinval", {1, 0}},
    {"svnapot", {1, 0}},
    {"svpbmt", {1, 0}},

    {"v", {1, 0}},

    // vendor-defined ('X') extensions
    {"xcvalu", {1, 0}},
    {"xcvbi", {1, 0}},
    {"xcvbitmanip", {1, 0}},
    {"xcvelw", {1, 0}},
    {"xcvmac", {1, 0}},
    {"xcvmem", {1, 0}},
    {"xcvsimd", {1, 0}},
    {"xsfcease", {1, 0}},
    {"xsfvcp", {1, 0}},
    {"xsfvfnrclipxfqf", {1, 0}},
    {"xsfvfwmaccqqq", {1, 0}},
    {"xsfvqmaccdod", {1, 0}},
    {"xsfvqmaccqoq", {1, 0}},
    {"xsifivecdiscarddlone", {1, 0}},
    {"xsifivecflushdlone", {1, 0}},
    {"xtheadba", {1, 0}},
    {"xtheadbb", {1, 0}},
    {"xtheadbs", {1, 0}},
    {"xtheadcmo", {1, 0}},
    {"xtheadcondmov", {1, 0}},
    {"xtheadfmemidx", {1, 0}},
    {"xtheadmac", {1, 0}},
    {"xtheadmemidx", {1, 0}},
    {"xtheadmempair", {1, 0}},
    {"xtheadsync", {1, 0}},
    {"xtheadvdot", {1, 0}},
    {"xventanacondops", {1, 0}},

    {"za128rs", {1, 0}},
    {"za64rs", {1, 0}},
    {"zacas", {1, 0}},
    {"zawrs", {1, 0}},

    {"zba", {1, 0}},
    {"zbb", {1, 0}},
    {"zbc", {1, 0}},
    {"zbkb", {1, 0}},
    {"zbkc", {1, 0}},
    {"zbkx", {1, 0}},
    {"zbs", {1, 0}},

    {"zca", {1, 0}},
    {"zcb", {1, 0}},
    {"zcd", {1, 0}},
    {"zce", {1, 0}},
    {"zcf", {1, 0}},
    {"zcmp", {1, 0}},
    {"zcmt", {1, 0}},

    {"zdinx", {1, 0}},

    {"zfa", {1, 0}},
    {"zfh", {1, 0}},
    {"zfhmin", {1, 0}},
    {"zfinx", {1, 0}},

    {"zhinx", {1, 0}},
    {"zhinxmin", {1, 0}},

    {"zic64b", {1, 0}},
    {"zicbom", {1, 0}},
    {"zicbop", {1, 0}},
    {"zicboz", {1, 0}},
    {"ziccamoa", {1, 0}},
    {"ziccif", {1, 0}},
    {"zicclsm", {1, 0}},
    {"ziccrse", {1, 0}},
    {"zicntr", {2, 0}},
    {"zicond", {1, 0}},
    {"zicsr", {2, 0}},
    {"zifencei", {2, 0}},
    {"zihintntl", {1, 0}},
    {"zihintpause", {2, 0}},
    {"zihpm", {2, 0}},

    {"zk", {1, 0}},
    {"zkn", {1, 0}},
    {"zknd", {1, 0}},
    {"zkne", {1, 0}},
    {"zknh", {1, 0}},
    {"zkr", {1, 0}},
    {"zks", {1, 0}},
    {"zksed", {1, 0}},
    {"zksh", {1, 0}},
    {"zkt", {1, 0}},

    {"zmmul", {1, 0}},

    {"zvbb", {1, 0}},
    {"zvbc", {1, 0}},

    {"zve32f", {1, 0}},
    {"zve32x", {1, 0}},
    {"zve64d", {1, 0}},
    {"zve64f", {1, 0}},
    {"zve64x", {1, 0}},

    {"zvfh", {1, 0}},
    {"zvfhmin", {1, 0}},

    // vector crypto
    {"zvkb", {1, 0}},
    {"zvkg", {1, 0}},
    {"zvkn", {1, 0}},
    {"zvknc", {1, 0}},
    {"zvkned", {1, 0}},
    {"zvkng", {1, 0}},
    {"zvknha", {1, 0}},
    {"zvknhb", {1, 0}},
    {"zvks", {1, 0}},
    {"zvksc", {1, 0}},
    {"zvksed", {1, 0}},
    {"zvksg", {1, 0}},
    {"zvksh", {1, 0}},
    {"zvkt", {1, 0}},

    {"zvl1024b", {1, 0}},
    {"zvl128b", {1, 0}},
    {"zvl16384b", {1, 0}},
    {"zvl2048b", {1, 0}},
    {"zvl256b", {1, 0}},
    {"zvl32768b", {1, 0}},
    {"zvl32b", {1, 0}},
    {"zvl4096b", {1, 0}},
    {"zvl512b", {1, 0}},
    {"zvl64b", {1, 0}},
    {"zvl65536b", {1, 0}},
    {"zvl8192b", {1, 0}},
};

// NOTE: This table should be sorted alphabetically by extension name.
// clang-format off
static const RISCVSupportedExtension SupportedExperimentalExtensions[] = {
    {"smmpm", {0, 8}},
    {"smnpm", {0, 8}},
    {"ssnpm", {0, 8}},
    {"sspm", {0, 8}},
    {"ssqosid", {1, 0}},
    {"supm", {0, 8}},

    {"zaamo", {0, 2}},
    {"zabha", {1, 0}},
    {"zalasr", {0, 1}},
    {"zalrsc", {0, 2}},

    {"zcmop", {0, 2}},

    {"zfbfmin", {1, 0}},

    {"zicfilp", {0, 4}},
    {"zicfiss", {0, 4}},

    {"zimop", {0, 1}},

    {"ztso", {0, 1}},

    {"zvfbfmin", {1, 0}},
    {"zvfbfwma", {1, 0}},
};
// clang-format on

static void verifyTables() {
#ifndef NDEBUG
  static std::atomic<bool> TableChecked(false);
  if (!TableChecked.load(std::memory_order_relaxed)) {
    assert(llvm::is_sorted(SupportedExtensions) &&
           "Extensions are not sorted by name");
    assert(llvm::is_sorted(SupportedExperimentalExtensions) &&
           "Experimental extensions are not sorted by name");
    TableChecked.store(true, std::memory_order_relaxed);
  }
#endif
}

static void PrintExtension(StringRef Name, StringRef Version,
                           StringRef Description) {
  outs().indent(4);
  unsigned VersionWidth = Description.empty() ? 0 : 10;
  outs() << left_justify(Name, 21) << left_justify(Version, VersionWidth)
         << Description << "\n";
}

void llvm::riscvExtensionsHelp(StringMap<StringRef> DescMap) {

  outs() << "All available -march extensions for RISC-V\n\n";
  PrintExtension("Name", "Version", (DescMap.empty() ? "" : "Description"));

  RISCVISAInfo::OrderedExtensionMap ExtMap;
  for (const auto &E : SupportedExtensions)
    ExtMap[E.Name] = {E.Version.Major, E.Version.Minor};
  for (const auto &E : ExtMap) {
    std::string Version =
        std::to_string(E.second.Major) + "." + std::to_string(E.second.Minor);
    PrintExtension(E.first, Version, DescMap[E.first]);
  }

  outs() << "\nExperimental extensions\n";
  ExtMap.clear();
  for (const auto &E : SupportedExperimentalExtensions)
    ExtMap[E.Name] = {E.Version.Major, E.Version.Minor};
  for (const auto &E : ExtMap) {
    std::string Version =
        std::to_string(E.second.Major) + "." + std::to_string(E.second.Minor);
    PrintExtension(E.first, Version, DescMap["experimental-" + E.first]);
  }

  outs() << "\nUse -march to specify the target's extension.\n"
            "For example, clang -march=rv32i_v1p0\n";
}

static bool stripExperimentalPrefix(StringRef &Ext) {
  return Ext.consume_front("experimental-");
}

// This function finds the last character that doesn't belong to a version
// (e.g. zba1p0 is extension 'zba' of version '1p0'). So the function will
// consume [0-9]*p[0-9]* starting from the backward. An extension name will not
// end with a digit or the letter 'p', so this function will parse correctly.
// NOTE: This function is NOT able to take empty strings or strings that only
// have version numbers and no extension name. It assumes the extension name
// will be at least more than one character.
static size_t findLastNonVersionCharacter(StringRef Ext) {
  assert(!Ext.empty() &&
         "Already guarded by if-statement in ::parseArchString");

  int Pos = Ext.size() - 1;
  while (Pos > 0 && isDigit(Ext[Pos]))
    Pos--;
  if (Pos > 0 && Ext[Pos] == 'p' && isDigit(Ext[Pos - 1])) {
    Pos--;
    while (Pos > 0 && isDigit(Ext[Pos]))
      Pos--;
  }
  return Pos;
}

namespace {
struct LessExtName {
  bool operator()(const RISCVSupportedExtension &LHS, StringRef RHS) {
    return StringRef(LHS.Name) < RHS;
  }
  bool operator()(StringRef LHS, const RISCVSupportedExtension &RHS) {
    return LHS < StringRef(RHS.Name);
  }
};
} // namespace

static std::optional<RISCVISAInfo::ExtensionVersion>
findDefaultVersion(StringRef ExtName) {
  // Find default version of an extension.
  // TODO: We might set default version based on profile or ISA spec.
  for (auto &ExtInfo : {ArrayRef(SupportedExtensions),
                        ArrayRef(SupportedExperimentalExtensions)}) {
    auto I = llvm::lower_bound(ExtInfo, ExtName, LessExtName());

    if (I == ExtInfo.end() || I->Name != ExtName)
      continue;

    return I->Version;
  }
  return std::nullopt;
}

void RISCVISAInfo::addExtension(StringRef ExtName,
                                RISCVISAInfo::ExtensionVersion Version) {
  Exts[ExtName.str()] = Version;
}

static StringRef getExtensionTypeDesc(StringRef Ext) {
  if (Ext.starts_with("s"))
    return "standard supervisor-level extension";
  if (Ext.starts_with("x"))
    return "non-standard user-level extension";
  if (Ext.starts_with("z"))
    return "standard user-level extension";
  return StringRef();
}

static StringRef getExtensionType(StringRef Ext) {
  if (Ext.starts_with("s"))
    return "s";
  if (Ext.starts_with("x"))
    return "x";
  if (Ext.starts_with("z"))
    return "z";
  return StringRef();
}

static std::optional<RISCVISAInfo::ExtensionVersion>
isExperimentalExtension(StringRef Ext) {
  auto I =
      llvm::lower_bound(SupportedExperimentalExtensions, Ext, LessExtName());
  if (I == std::end(SupportedExperimentalExtensions) || I->Name != Ext)
    return std::nullopt;

  return I->Version;
}

bool RISCVISAInfo::isSupportedExtensionFeature(StringRef Ext) {
  bool IsExperimental = stripExperimentalPrefix(Ext);

  ArrayRef<RISCVSupportedExtension> ExtInfo =
      IsExperimental ? ArrayRef(SupportedExperimentalExtensions)
                     : ArrayRef(SupportedExtensions);

  auto I = llvm::lower_bound(ExtInfo, Ext, LessExtName());
  return I != ExtInfo.end() && I->Name == Ext;
}

bool RISCVISAInfo::isSupportedExtension(StringRef Ext) {
  verifyTables();

  for (auto ExtInfo : {ArrayRef(SupportedExtensions),
                       ArrayRef(SupportedExperimentalExtensions)}) {
    auto I = llvm::lower_bound(ExtInfo, Ext, LessExtName());
    if (I != ExtInfo.end() && I->Name == Ext)
      return true;
  }

  return false;
}

bool RISCVISAInfo::isSupportedExtension(StringRef Ext, unsigned MajorVersion,
                                        unsigned MinorVersion) {
  for (auto ExtInfo : {ArrayRef(SupportedExtensions),
                       ArrayRef(SupportedExperimentalExtensions)}) {
    auto Range =
        std::equal_range(ExtInfo.begin(), ExtInfo.end(), Ext, LessExtName());
    for (auto I = Range.first, E = Range.second; I != E; ++I)
      if (I->Version.Major == MajorVersion && I->Version.Minor == MinorVersion)
        return true;
  }

  return false;
}

bool RISCVISAInfo::hasExtension(StringRef Ext) const {
  stripExperimentalPrefix(Ext);

  if (!isSupportedExtension(Ext))
    return false;

  return Exts.count(Ext.str()) != 0;
}

// We rank extensions in the following order:
// -Single letter extensions in canonical order.
// -Unknown single letter extensions in alphabetical order.
// -Multi-letter extensions starting with 'z' sorted by canonical order of
//  the second letter then sorted alphabetically.
// -Multi-letter extensions starting with 's' in alphabetical order.
// -(TODO) Multi-letter extensions starting with 'zxm' in alphabetical order.
// -X extensions in alphabetical order.
// These flags are used to indicate the category. The first 6 bits store the
// single letter extension rank for single letter and multi-letter extensions
// starting with 'z'.
enum RankFlags {
  RF_Z_EXTENSION = 1 << 6,
  RF_S_EXTENSION = 1 << 7,
  RF_X_EXTENSION = 1 << 8,
};

// Get the rank for single-letter extension, lower value meaning higher
// priority.
static unsigned singleLetterExtensionRank(char Ext) {
  assert(Ext >= 'a' && Ext <= 'z');
  switch (Ext) {
  case 'i':
    return 0;
  case 'e':
    return 1;
  }

  size_t Pos = AllStdExts.find(Ext);
  if (Pos != StringRef::npos)
    return Pos + 2; // Skip 'e' and 'i' from above.

  // If we got an unknown extension letter, then give it an alphabetical
  // order, but after all known standard extensions.
  return 2 + AllStdExts.size() + (Ext - 'a');
}

// Get the rank for multi-letter extension, lower value meaning higher
// priority/order in canonical order.
static unsigned getExtensionRank(const std::string &ExtName) {
  assert(ExtName.size() >= 1);
  switch (ExtName[0]) {
  case 's':
    return RF_S_EXTENSION;
  case 'z':
    assert(ExtName.size() >= 2);
    // `z` extension must be sorted by canonical order of second letter.
    // e.g. zmx has higher rank than zax.
    return RF_Z_EXTENSION | singleLetterExtensionRank(ExtName[1]);
  case 'x':
    return RF_X_EXTENSION;
  default:
    assert(ExtName.size() == 1);
    return singleLetterExtensionRank(ExtName[0]);
  }
}

// Compare function for extension.
// Only compare the extension name, ignore version comparison.
bool RISCVISAInfo::compareExtension(const std::string &LHS,
                                    const std::string &RHS) {
  unsigned LHSRank = getExtensionRank(LHS);
  unsigned RHSRank = getExtensionRank(RHS);

  // If the ranks differ, pick the lower rank.
  if (LHSRank != RHSRank)
    return LHSRank < RHSRank;

  // If the rank is same, it must be sorted by lexicographic order.
  return LHS < RHS;
}

std::vector<std::string> RISCVISAInfo::toFeatures(bool AddAllExtensions,
                                                  bool IgnoreUnknown) const {
  std::vector<std::string> Features;
  for (const auto &[ExtName, _] : Exts) {
    // i is a base instruction set, not an extension (see
    // https://github.com/riscv/riscv-isa-manual/blob/main/src/naming.adoc#base-integer-isa)
    // and is not recognized in clang -cc1
    if (ExtName == "i")
      continue;
    if (IgnoreUnknown && !isSupportedExtension(ExtName))
      continue;

    if (isExperimentalExtension(ExtName)) {
      Features.push_back((llvm::Twine("+experimental-") + ExtName).str());
    } else {
      Features.push_back((llvm::Twine("+") + ExtName).str());
    }
  }
  if (AddAllExtensions) {
    for (const RISCVSupportedExtension &Ext : SupportedExtensions) {
      if (Exts.count(Ext.Name))
        continue;
      Features.push_back((llvm::Twine("-") + Ext.Name).str());
    }

    for (const RISCVSupportedExtension &Ext : SupportedExperimentalExtensions) {
      if (Exts.count(Ext.Name))
        continue;
      Features.push_back((llvm::Twine("-experimental-") + Ext.Name).str());
    }
  }
  return Features;
}

static Error getStringErrorForInvalidExt(std::string_view ExtName) {
  if (ExtName.size() == 1) {
    return createStringError(errc::invalid_argument,
                             "unsupported standard user-level extension '" +
                                 ExtName + "'");
  }
  return createStringError(errc::invalid_argument,
                           "unsupported " + getExtensionTypeDesc(ExtName) +
                               " '" + ExtName + "'");
}

// Extensions may have a version number, and may be separated by
// an underscore '_' e.g.: rv32i2_m2.
// Version number is divided into major and minor version numbers,
// separated by a 'p'. If the minor version is 0 then 'p0' can be
// omitted from the version string. E.g., rv32i2p0, rv32i2, rv32i2p1.
static Error getExtensionVersion(StringRef Ext, StringRef In, unsigned &Major,
                                 unsigned &Minor, unsigned &ConsumeLength,
                                 bool EnableExperimentalExtension,
                                 bool ExperimentalExtensionVersionCheck) {
  StringRef MajorStr, MinorStr;
  Major = 0;
  Minor = 0;
  ConsumeLength = 0;
  MajorStr = In.take_while(isDigit);
  In = In.substr(MajorStr.size());

  if (!MajorStr.empty() && In.consume_front("p")) {
    MinorStr = In.take_while(isDigit);
    In = In.substr(MajorStr.size() + MinorStr.size() - 1);

    // Expected 'p' to be followed by minor version number.
    if (MinorStr.empty()) {
      return createStringError(
          errc::invalid_argument,
          "minor version number missing after 'p' for extension '" + Ext + "'");
    }
  }

  if (!MajorStr.empty() && MajorStr.getAsInteger(10, Major))
    return createStringError(
        errc::invalid_argument,
        "Failed to parse major version number for extension '" + Ext + "'");

  if (!MinorStr.empty() && MinorStr.getAsInteger(10, Minor))
    return createStringError(
        errc::invalid_argument,
        "Failed to parse minor version number for extension '" + Ext + "'");

  ConsumeLength = MajorStr.size();

  if (!MinorStr.empty())
    ConsumeLength += MinorStr.size() + 1 /*'p'*/;

  // Expected multi-character extension with version number to have no
  // subsequent characters (i.e. must either end string or be followed by
  // an underscore).
  if (Ext.size() > 1 && In.size())
    return createStringError(
        errc::invalid_argument,
        "multi-character extensions must be separated by underscores");

  // If experimental extension, require use of current version number
  if (auto ExperimentalExtension = isExperimentalExtension(Ext)) {
    if (!EnableExperimentalExtension)
      return createStringError(errc::invalid_argument,
                               "requires '-menable-experimental-extensions' "
                               "for experimental extension '" +
                                   Ext + "'");

    if (ExperimentalExtensionVersionCheck &&
        (MajorStr.empty() && MinorStr.empty()))
      return createStringError(
          errc::invalid_argument,
          "experimental extension requires explicit version number `" + Ext +
              "`");

    auto SupportedVers = *ExperimentalExtension;
    if (ExperimentalExtensionVersionCheck &&
        (Major != SupportedVers.Major || Minor != SupportedVers.Minor)) {
      std::string Error = "unsupported version number " + MajorStr.str();
      if (!MinorStr.empty())
        Error += "." + MinorStr.str();
      Error += " for experimental extension '" + Ext.str() +
               "' (this compiler supports " + utostr(SupportedVers.Major) +
               "." + utostr(SupportedVers.Minor) + ")";
      return createStringError(errc::invalid_argument, Error);
    }
    return Error::success();
  }

  // Exception rule for `g`, we don't have clear version scheme for that on
  // ISA spec.
  if (Ext == "g")
    return Error::success();

  if (MajorStr.empty() && MinorStr.empty()) {
    if (auto DefaultVersion = findDefaultVersion(Ext)) {
      Major = DefaultVersion->Major;
      Minor = DefaultVersion->Minor;
    }
    // No matter found or not, return success, assume other place will
    // verify.
    return Error::success();
  }

  if (RISCVISAInfo::isSupportedExtension(Ext, Major, Minor))
    return Error::success();

  if (!RISCVISAInfo::isSupportedExtension(Ext))
    return getStringErrorForInvalidExt(Ext);

  std::string Error = "unsupported version number " + std::string(MajorStr);
  if (!MinorStr.empty())
    Error += "." + MinorStr.str();
  Error += " for extension '" + Ext.str() + "'";
  return createStringError(errc::invalid_argument, Error);
}

llvm::Expected<std::unique_ptr<RISCVISAInfo>>
RISCVISAInfo::parseFeatures(unsigned XLen,
                            const std::vector<std::string> &Features) {
  assert(XLen == 32 || XLen == 64);
  std::unique_ptr<RISCVISAInfo> ISAInfo(new RISCVISAInfo(XLen));

  for (auto &Feature : Features) {
    StringRef ExtName = Feature;
    bool Experimental = false;
    assert(ExtName.size() > 1 && (ExtName[0] == '+' || ExtName[0] == '-'));
    bool Add = ExtName[0] == '+';
    ExtName = ExtName.drop_front(1); // Drop '+' or '-'
    Experimental = stripExperimentalPrefix(ExtName);
    auto ExtensionInfos = Experimental
                              ? ArrayRef(SupportedExperimentalExtensions)
                              : ArrayRef(SupportedExtensions);
    auto ExtensionInfoIterator =
        llvm::lower_bound(ExtensionInfos, ExtName, LessExtName());

    // Not all features is related to ISA extension, like `relax` or
    // `save-restore`, skip those feature.
    if (ExtensionInfoIterator == ExtensionInfos.end() ||
        ExtensionInfoIterator->Name != ExtName)
      continue;

    if (Add)
      ISAInfo->addExtension(ExtName, ExtensionInfoIterator->Version);
    else
      ISAInfo->Exts.erase(ExtName.str());
  }

  return RISCVISAInfo::postProcessAndChecking(std::move(ISAInfo));
}

llvm::Expected<std::unique_ptr<RISCVISAInfo>>
RISCVISAInfo::parseNormalizedArchString(StringRef Arch) {
  if (llvm::any_of(Arch, isupper)) {
    return createStringError(errc::invalid_argument,
                             "string must be lowercase");
  }
  // Must start with a valid base ISA name.
  unsigned XLen;
  if (Arch.starts_with("rv32i") || Arch.starts_with("rv32e"))
    XLen = 32;
  else if (Arch.starts_with("rv64i") || Arch.starts_with("rv64e"))
    XLen = 64;
  else
    return createStringError(errc::invalid_argument,
                             "arch string must begin with valid base ISA");
  std::unique_ptr<RISCVISAInfo> ISAInfo(new RISCVISAInfo(XLen));
  // Discard rv32/rv64 prefix.
  Arch = Arch.substr(4);

  // Each extension is of the form ${name}${major_version}p${minor_version}
  // and separated by _. Split by _ and then extract the name and version
  // information for each extension.
  SmallVector<StringRef, 8> Split;
  Arch.split(Split, '_');
  for (StringRef Ext : Split) {
    StringRef Prefix, MinorVersionStr;
    std::tie(Prefix, MinorVersionStr) = Ext.rsplit('p');
    if (MinorVersionStr.empty())
      return createStringError(errc::invalid_argument,
                               "extension lacks version in expected format");
    unsigned MajorVersion, MinorVersion;
    if (MinorVersionStr.getAsInteger(10, MinorVersion))
      return createStringError(errc::invalid_argument,
                               "failed to parse minor version number");

    // Split Prefix into the extension name and the major version number
    // (the trailing digits of Prefix).
    int TrailingDigits = 0;
    StringRef ExtName = Prefix;
    while (!ExtName.empty()) {
      if (!isDigit(ExtName.back()))
        break;
      ExtName = ExtName.drop_back(1);
      TrailingDigits++;
    }
    if (!TrailingDigits)
      return createStringError(errc::invalid_argument,
                               "extension lacks version in expected format");

    StringRef MajorVersionStr = Prefix.take_back(TrailingDigits);
    if (MajorVersionStr.getAsInteger(10, MajorVersion))
      return createStringError(errc::invalid_argument,
                               "failed to parse major version number");
    ISAInfo->addExtension(ExtName, {MajorVersion, MinorVersion});
  }
  ISAInfo->updateFLen();
  ISAInfo->updateMinVLen();
  ISAInfo->updateMaxELen();
  return std::move(ISAInfo);
}

static Error splitExtsByUnderscore(StringRef Exts,
                                   std::vector<std::string> &SplitExts) {
  SmallVector<StringRef, 8> Split;
  if (Exts.empty())
    return Error::success();

  Exts.split(Split, "_");

  for (auto Ext : Split) {
    if (Ext.empty())
      return createStringError(errc::invalid_argument,
                               "extension name missing after separator '_'");

    SplitExts.push_back(Ext.str());
  }
  return Error::success();
}

static Error processMultiLetterExtension(
    StringRef RawExt,
    MapVector<std::string, RISCVISAInfo::ExtensionVersion,
              std::map<std::string, unsigned>> &SeenExtMap,
    bool IgnoreUnknown, bool EnableExperimentalExtension,
    bool ExperimentalExtensionVersionCheck) {
  StringRef Type = getExtensionType(RawExt);
  StringRef Desc = getExtensionTypeDesc(RawExt);
  auto Pos = findLastNonVersionCharacter(RawExt) + 1;
  StringRef Name(RawExt.substr(0, Pos));
  StringRef Vers(RawExt.substr(Pos));

  if (Type.empty()) {
    if (IgnoreUnknown)
      return Error::success();
    return createStringError(errc::invalid_argument,
                             "invalid extension prefix '" + RawExt + "'");
  }

  if (!IgnoreUnknown && Name.size() == Type.size())
    return createStringError(errc::invalid_argument,
                             Desc + " name missing after '" + Type + "'");

  unsigned Major, Minor, ConsumeLength;
  if (auto E = getExtensionVersion(Name, Vers, Major, Minor, ConsumeLength,
                                   EnableExperimentalExtension,
                                   ExperimentalExtensionVersionCheck)) {
    if (IgnoreUnknown) {
      consumeError(std::move(E));
      return Error::success();
    }
    return E;
  }

  // Check if duplicated extension.
  if (!IgnoreUnknown && SeenExtMap.contains(Name.str()))
    return createStringError(errc::invalid_argument,
                             "duplicated " + Desc + " '" + Name + "'");

  if (IgnoreUnknown && !RISCVISAInfo::isSupportedExtension(Name))
    return Error::success();

  SeenExtMap[Name.str()] = {Major, Minor};
  return Error::success();
}

static Error processSingleLetterExtension(
    StringRef &RawExt,
    MapVector<std::string, RISCVISAInfo::ExtensionVersion,
              std::map<std::string, unsigned>> &SeenExtMap,
    bool IgnoreUnknown, bool EnableExperimentalExtension,
    bool ExperimentalExtensionVersionCheck) {
  unsigned Major, Minor, ConsumeLength;
  StringRef Name = RawExt.take_front(1);
  RawExt.consume_front(Name);
  if (auto E = getExtensionVersion(Name, RawExt, Major, Minor, ConsumeLength,
                                   EnableExperimentalExtension,
                                   ExperimentalExtensionVersionCheck)) {
    if (IgnoreUnknown) {
      consumeError(std::move(E));
      RawExt = RawExt.substr(ConsumeLength);
      return Error::success();
    }
    return E;
  }

  RawExt = RawExt.substr(ConsumeLength);

  // Check if duplicated extension.
  if (!IgnoreUnknown && SeenExtMap.contains(Name.str()))
    return createStringError(errc::invalid_argument,
                             "duplicated standard user-level extension '" +
                                 Name + "'");

  if (IgnoreUnknown && !RISCVISAInfo::isSupportedExtension(Name))
    return Error::success();

  SeenExtMap[Name.str()] = {Major, Minor};
  return Error::success();
}

llvm::Expected<std::unique_ptr<RISCVISAInfo>>
RISCVISAInfo::parseArchString(StringRef Arch, bool EnableExperimentalExtension,
                              bool ExperimentalExtensionVersionCheck,
                              bool IgnoreUnknown) {
  // RISC-V ISA strings must be lowercase.
  if (llvm::any_of(Arch, isupper)) {
    return createStringError(errc::invalid_argument,
                             "string must be lowercase");
  }

  bool HasRV64 = Arch.starts_with("rv64");
  // ISA string must begin with rv32 or rv64.
  if (!(Arch.starts_with("rv32") || HasRV64) || (Arch.size() < 5)) {
    return createStringError(
        errc::invalid_argument,
        "string must begin with rv32{i,e,g} or rv64{i,e,g}");
  }

  unsigned XLen = HasRV64 ? 64 : 32;
  std::unique_ptr<RISCVISAInfo> ISAInfo(new RISCVISAInfo(XLen));
  MapVector<std::string, RISCVISAInfo::ExtensionVersion,
            std::map<std::string, unsigned>>
      SeenExtMap;

  // The canonical order specified in ISA manual.
  // Ref: Table 22.1 in RISC-V User-Level ISA V2.2
  StringRef StdExts = AllStdExts;
  char Baseline = Arch[4];

  // First letter should be 'e', 'i' or 'g'.
  switch (Baseline) {
  default:
    return createStringError(errc::invalid_argument,
                             "first letter should be 'e', 'i' or 'g'");
  case 'e':
  case 'i':
    break;
  case 'g':
    // g expands to extensions in RISCVGImplications.
    if (Arch.size() > 5 && isDigit(Arch[5]))
      return createStringError(errc::invalid_argument,
                               "version not supported for 'g'");
    StdExts = StdExts.drop_front(4);
    break;
  }

  if (Arch.back() == '_')
    return createStringError(errc::invalid_argument,
                             "extension name missing after separator '_'");

  // Skip rvxxx
  StringRef Exts = Arch.substr(5);

  unsigned Major, Minor, ConsumeLength;
  if (Baseline == 'g') {
    // Versions for g are disallowed, and this was checked for previously.
    ConsumeLength = 0;

    // No matter which version is given to `g`, we always set imafd to default
    // version since the we don't have clear version scheme for that on
    // ISA spec.
    for (const auto *Ext : RISCVGImplications) {
      if (auto Version = findDefaultVersion(Ext)) {
        // Postpone AddExtension until end of this function
        SeenExtMap[Ext] = {Version->Major, Version->Minor};
      } else
        llvm_unreachable("Default extension version not found?");
    }
  } else {
    // Baseline is `i` or `e`
    if (auto E = getExtensionVersion(
            StringRef(&Baseline, 1), Exts, Major, Minor, ConsumeLength,
            EnableExperimentalExtension, ExperimentalExtensionVersionCheck)) {
      if (!IgnoreUnknown)
        return std::move(E);
      // If IgnoreUnknown, then ignore an unrecognised version of the baseline
      // ISA and just use the default supported version.
      consumeError(std::move(E));
      auto Version = findDefaultVersion(StringRef(&Baseline, 1));
      Major = Version->Major;
      Minor = Version->Minor;
    }

    // Postpone AddExtension until end of this function
    SeenExtMap[StringRef(&Baseline, 1).str()] = {Major, Minor};
  }

  // Consume the base ISA version number and any '_' between rvxxx and the
  // first extension
  Exts = Exts.drop_front(ConsumeLength);
  Exts.consume_front("_");

  std::vector<std::string> SplittedExts;
  if (auto E = splitExtsByUnderscore(Exts, SplittedExts))
    return std::move(E);

  for (auto &Ext : SplittedExts) {
    StringRef CurrExt = Ext;
    while (!CurrExt.empty()) {
      if (AllStdExts.contains(CurrExt.front())) {
        if (auto E = processSingleLetterExtension(
                CurrExt, SeenExtMap, IgnoreUnknown, EnableExperimentalExtension,
                ExperimentalExtensionVersionCheck))
          return std::move(E);
      } else if (CurrExt.front() == 'z' || CurrExt.front() == 's' ||
                 CurrExt.front() == 'x') {
        // Handle other types of extensions other than the standard
        // general purpose and standard user-level extensions.
        // Parse the ISA string containing non-standard user-level
        // extensions, standard supervisor-level extensions and
        // non-standard supervisor-level extensions.
        // These extensions start with 'z', 's', 'x' prefixes, might have a
        // version number (major, minor) and are separated by a single
        // underscore '_'. We do not enforce a canonical order for them.
        if (auto E = processMultiLetterExtension(
                CurrExt, SeenExtMap, IgnoreUnknown, EnableExperimentalExtension,
                ExperimentalExtensionVersionCheck))
          return std::move(E);
        // Multi-letter extension must be seperate following extension with
        // underscore
        break;
      } else {
        // FIXME: Could it be ignored by IgnoreUnknown?
        return createStringError(errc::invalid_argument,
                                 "invalid standard user-level extension '" +
                                     Twine(CurrExt.front()) + "'");
      }
    }
  }

  // Check all Extensions are supported.
  for (auto &SeenExtAndVers : SeenExtMap) {
    const std::string &ExtName = SeenExtAndVers.first;
    RISCVISAInfo::ExtensionVersion ExtVers = SeenExtAndVers.second;

    if (!RISCVISAInfo::isSupportedExtension(ExtName))
      return getStringErrorForInvalidExt(ExtName);
    ISAInfo->addExtension(ExtName, ExtVers);
  }

  return RISCVISAInfo::postProcessAndChecking(std::move(ISAInfo));
}

Error RISCVISAInfo::checkDependency() {
  bool HasC = Exts.count("c") != 0;
  bool HasF = Exts.count("f") != 0;
  bool HasZfinx = Exts.count("zfinx") != 0;
  bool HasVector = Exts.count("zve32x") != 0;
  bool HasZvl = MinVLen != 0;
  bool HasZcmt = Exts.count("zcmt") != 0;

  if (HasF && HasZfinx)
    return createStringError(errc::invalid_argument,
                             "'f' and 'zfinx' extensions are incompatible");

  if (HasZvl && !HasVector)
    return createStringError(
        errc::invalid_argument,
        "'zvl*b' requires 'v' or 'zve*' extension to also be specified");

  if (Exts.count("zvbb") && !HasVector)
    return createStringError(
        errc::invalid_argument,
        "'zvbb' requires 'v' or 'zve*' extension to also be specified");

  if (Exts.count("zvbc") && !Exts.count("zve64x"))
    return createStringError(
        errc::invalid_argument,
        "'zvbc' requires 'v' or 'zve64*' extension to also be specified");

  if ((Exts.count("zvkb") || Exts.count("zvkg") || Exts.count("zvkned") ||
       Exts.count("zvknha") || Exts.count("zvksed") || Exts.count("zvksh")) &&
      !HasVector)
    return createStringError(
        errc::invalid_argument,
        "'zvk*' requires 'v' or 'zve*' extension to also be specified");

  if (Exts.count("zvknhb") && !Exts.count("zve64x"))
    return createStringError(
        errc::invalid_argument,
        "'zvknhb' requires 'v' or 'zve64*' extension to also be specified");

  if ((HasZcmt || Exts.count("zcmp")) && Exts.count("d") &&
      (HasC || Exts.count("zcd")))
    return createStringError(
        errc::invalid_argument,
        Twine("'") + (HasZcmt ? "zcmt" : "zcmp") +
        "' extension is incompatible with '" + (HasC ? "c" : "zcd") +
        "' extension when 'd' extension is enabled");

  if (XLen != 32 && Exts.count("zcf"))
    return createStringError(errc::invalid_argument,
                             "'zcf' is only supported for 'rv32'");

  return Error::success();
}

static const char *ImpliedExtsD[] = {"f"};
static const char *ImpliedExtsF[] = {"zicsr"};
static const char *ImpliedExtsV[] = {"zvl128b", "zve64d"};
static const char *ImpliedExtsXTHeadVdot[] = {"v"};
static const char *ImpliedExtsXSfvcp[] = {"zve32x"};
static const char *ImpliedExtsXSfvfnrclipxfqf[] = {"zve32f"};
static const char *ImpliedExtsXSfvfwmaccqqq[] = {"zvfbfmin"};
static const char *ImpliedExtsXSfvqmaccdod[] = {"zve32x"};
static const char *ImpliedExtsXSfvqmaccqoq[] = {"zve32x"};
static const char *ImpliedExtsZabha[] = {"a"};
static const char *ImpliedExtsZacas[] = {"a"};
static const char *ImpliedExtsZcb[] = {"zca"};
static const char *ImpliedExtsZcd[] = {"d", "zca"};
static const char *ImpliedExtsZce[] = {"zcb", "zcmp", "zcmt"};
static const char *ImpliedExtsZcf[] = {"f", "zca"};
static const char *ImpliedExtsZcmop[] = {"zca"};
static const char *ImpliedExtsZcmp[] = {"zca"};
static const char *ImpliedExtsZcmt[] = {"zca", "zicsr"};
static const char *ImpliedExtsZdinx[] = {"zfinx"};
static const char *ImpliedExtsZfa[] = {"f"};
static const char *ImpliedExtsZfbfmin[] = {"f"};
static const char *ImpliedExtsZfh[] = {"zfhmin"};
static const char *ImpliedExtsZfhmin[] = {"f"};
static const char *ImpliedExtsZfinx[] = {"zicsr"};
static const char *ImpliedExtsZhinx[] = {"zhinxmin"};
static const char *ImpliedExtsZhinxmin[] = {"zfinx"};
static const char *ImpliedExtsZicntr[] = {"zicsr"};
static const char *ImpliedExtsZicfiss[] = {"zicsr", "zimop"};
static const char *ImpliedExtsZihpm[] = {"zicsr"};
static const char *ImpliedExtsZk[] = {"zkn", "zkt", "zkr"};
static const char *ImpliedExtsZkn[] = {"zbkb", "zbkc", "zbkx",
                                       "zkne", "zknd", "zknh"};
static const char *ImpliedExtsZks[] = {"zbkb", "zbkc", "zbkx", "zksed", "zksh"};
static const char *ImpliedExtsZvbb[] = {"zvkb"};
static const char *ImpliedExtsZve32f[] = {"zve32x", "f"};
static const char *ImpliedExtsZve32x[] = {"zvl32b", "zicsr"};
static const char *ImpliedExtsZve64d[] = {"zve64f", "d"};
static const char *ImpliedExtsZve64f[] = {"zve64x", "zve32f"};
static const char *ImpliedExtsZve64x[] = {"zve32x", "zvl64b"};
static const char *ImpliedExtsZvfbfmin[] = {"zve32f"};
static const char *ImpliedExtsZvfbfwma[] = {"zvfbfmin", "zfbfmin"};
static const char *ImpliedExtsZvfh[] = {"zvfhmin", "zfhmin"};
static const char *ImpliedExtsZvfhmin[] = {"zve32f"};
static const char *ImpliedExtsZvkn[] = {"zvkb", "zvkned", "zvknhb", "zvkt"};
static const char *ImpliedExtsZvknc[] = {"zvbc", "zvkn"};
static const char *ImpliedExtsZvkng[] = {"zvkg", "zvkn"};
static const char *ImpliedExtsZvknhb[] = {"zve64x"};
static const char *ImpliedExtsZvks[] = {"zvkb", "zvksed", "zvksh", "zvkt"};
static const char *ImpliedExtsZvksc[] = {"zvbc", "zvks"};
static const char *ImpliedExtsZvksg[] = {"zvkg", "zvks"};
static const char *ImpliedExtsZvl1024b[] = {"zvl512b"};
static const char *ImpliedExtsZvl128b[] = {"zvl64b"};
static const char *ImpliedExtsZvl16384b[] = {"zvl8192b"};
static const char *ImpliedExtsZvl2048b[] = {"zvl1024b"};
static const char *ImpliedExtsZvl256b[] = {"zvl128b"};
static const char *ImpliedExtsZvl32768b[] = {"zvl16384b"};
static const char *ImpliedExtsZvl4096b[] = {"zvl2048b"};
static const char *ImpliedExtsZvl512b[] = {"zvl256b"};
static const char *ImpliedExtsZvl64b[] = {"zvl32b"};
static const char *ImpliedExtsZvl65536b[] = {"zvl32768b"};
static const char *ImpliedExtsZvl8192b[] = {"zvl4096b"};

struct ImpliedExtsEntry {
  StringLiteral Name;
  ArrayRef<const char *> Exts;

  bool operator<(const ImpliedExtsEntry &Other) const {
    return Name < Other.Name;
  }

  bool operator<(StringRef Other) const { return Name < Other; }
};

// Note: The table needs to be sorted by name.
static constexpr ImpliedExtsEntry ImpliedExts[] = {
    {{"d"}, {ImpliedExtsD}},
    {{"f"}, {ImpliedExtsF}},
    {{"v"}, {ImpliedExtsV}},
    {{"xsfvcp"}, {ImpliedExtsXSfvcp}},
    {{"xsfvfnrclipxfqf"}, {ImpliedExtsXSfvfnrclipxfqf}},
    {{"xsfvfwmaccqqq"}, {ImpliedExtsXSfvfwmaccqqq}},
    {{"xsfvqmaccdod"}, {ImpliedExtsXSfvqmaccdod}},
    {{"xsfvqmaccqoq"}, {ImpliedExtsXSfvqmaccqoq}},
    {{"xtheadvdot"}, {ImpliedExtsXTHeadVdot}},
    {{"zabha"}, {ImpliedExtsZabha}},
    {{"zacas"}, {ImpliedExtsZacas}},
    {{"zcb"}, {ImpliedExtsZcb}},
    {{"zcd"}, {ImpliedExtsZcd}},
    {{"zce"}, {ImpliedExtsZce}},
    {{"zcf"}, {ImpliedExtsZcf}},
    {{"zcmop"}, {ImpliedExtsZcmop}},
    {{"zcmp"}, {ImpliedExtsZcmp}},
    {{"zcmt"}, {ImpliedExtsZcmt}},
    {{"zdinx"}, {ImpliedExtsZdinx}},
    {{"zfa"}, {ImpliedExtsZfa}},
    {{"zfbfmin"}, {ImpliedExtsZfbfmin}},
    {{"zfh"}, {ImpliedExtsZfh}},
    {{"zfhmin"}, {ImpliedExtsZfhmin}},
    {{"zfinx"}, {ImpliedExtsZfinx}},
    {{"zhinx"}, {ImpliedExtsZhinx}},
    {{"zhinxmin"}, {ImpliedExtsZhinxmin}},
    {{"zicfiss"}, {ImpliedExtsZicfiss}},
    {{"zicntr"}, {ImpliedExtsZicntr}},
    {{"zihpm"}, {ImpliedExtsZihpm}},
    {{"zk"}, {ImpliedExtsZk}},
    {{"zkn"}, {ImpliedExtsZkn}},
    {{"zks"}, {ImpliedExtsZks}},
    {{"zvbb"}, {ImpliedExtsZvbb}},
    {{"zve32f"}, {ImpliedExtsZve32f}},
    {{"zve32x"}, {ImpliedExtsZve32x}},
    {{"zve64d"}, {ImpliedExtsZve64d}},
    {{"zve64f"}, {ImpliedExtsZve64f}},
    {{"zve64x"}, {ImpliedExtsZve64x}},
    {{"zvfbfmin"}, {ImpliedExtsZvfbfmin}},
    {{"zvfbfwma"}, {ImpliedExtsZvfbfwma}},
    {{"zvfh"}, {ImpliedExtsZvfh}},
    {{"zvfhmin"}, {ImpliedExtsZvfhmin}},
    {{"zvkn"}, {ImpliedExtsZvkn}},
    {{"zvknc"}, {ImpliedExtsZvknc}},
    {{"zvkng"}, {ImpliedExtsZvkng}},
    {{"zvknhb"}, {ImpliedExtsZvknhb}},
    {{"zvks"}, {ImpliedExtsZvks}},
    {{"zvksc"}, {ImpliedExtsZvksc}},
    {{"zvksg"}, {ImpliedExtsZvksg}},
    {{"zvl1024b"}, {ImpliedExtsZvl1024b}},
    {{"zvl128b"}, {ImpliedExtsZvl128b}},
    {{"zvl16384b"}, {ImpliedExtsZvl16384b}},
    {{"zvl2048b"}, {ImpliedExtsZvl2048b}},
    {{"zvl256b"}, {ImpliedExtsZvl256b}},
    {{"zvl32768b"}, {ImpliedExtsZvl32768b}},
    {{"zvl4096b"}, {ImpliedExtsZvl4096b}},
    {{"zvl512b"}, {ImpliedExtsZvl512b}},
    {{"zvl64b"}, {ImpliedExtsZvl64b}},
    {{"zvl65536b"}, {ImpliedExtsZvl65536b}},
    {{"zvl8192b"}, {ImpliedExtsZvl8192b}},
};

void RISCVISAInfo::updateImplication() {
  bool HasE = Exts.count("e") != 0;
  bool HasI = Exts.count("i") != 0;

  // If not in e extension and i extension does not exist, i extension is
  // implied
  if (!HasE && !HasI) {
    auto Version = findDefaultVersion("i");
    addExtension("i", Version.value());
  }

  assert(llvm::is_sorted(ImpliedExts) && "Table not sorted by Name");

  // This loop may execute over 1 iteration since implication can be layered
  // Exits loop if no more implication is applied
  SmallSetVector<StringRef, 16> WorkList;
  for (auto const &Ext : Exts)
    WorkList.insert(Ext.first);

  while (!WorkList.empty()) {
    StringRef ExtName = WorkList.pop_back_val();
    auto I = llvm::lower_bound(ImpliedExts, ExtName);
    if (I != std::end(ImpliedExts) && I->Name == ExtName) {
      for (const char *ImpliedExt : I->Exts) {
        if (WorkList.count(ImpliedExt))
          continue;
        if (Exts.count(ImpliedExt))
          continue;
        auto Version = findDefaultVersion(ImpliedExt);
        addExtension(ImpliedExt, Version.value());
        WorkList.insert(ImpliedExt);
      }
    }
  }

  // Add Zcf if Zce and F are enabled on RV32.
  if (XLen == 32 && Exts.count("zce") && Exts.count("f") &&
      !Exts.count("zcf")) {
    auto Version = findDefaultVersion("zcf");
    addExtension("zcf", Version.value());
  }
}

struct CombinedExtsEntry {
  StringLiteral CombineExt;
  ArrayRef<const char *> RequiredExts;
};

static constexpr CombinedExtsEntry CombineIntoExts[] = {
    {{"zk"}, {ImpliedExtsZk}},
    {{"zkn"}, {ImpliedExtsZkn}},
    {{"zks"}, {ImpliedExtsZks}},
    {{"zvkn"}, {ImpliedExtsZvkn}},
    {{"zvknc"}, {ImpliedExtsZvknc}},
    {{"zvkng"}, {ImpliedExtsZvkng}},
    {{"zvks"}, {ImpliedExtsZvks}},
    {{"zvksc"}, {ImpliedExtsZvksc}},
    {{"zvksg"}, {ImpliedExtsZvksg}},
};

void RISCVISAInfo::updateCombination() {
  bool IsNewCombine = false;
  do {
    IsNewCombine = false;
    for (CombinedExtsEntry CombineIntoExt : CombineIntoExts) {
      auto CombineExt = CombineIntoExt.CombineExt;
      auto RequiredExts = CombineIntoExt.RequiredExts;
      if (hasExtension(CombineExt))
        continue;
      bool IsAllRequiredFeatureExist = true;
      for (const char *Ext : RequiredExts)
        IsAllRequiredFeatureExist &= hasExtension(Ext);
      if (IsAllRequiredFeatureExist) {
        auto Version = findDefaultVersion(CombineExt);
        addExtension(CombineExt, Version.value());
        IsNewCombine = true;
      }
    }
  } while (IsNewCombine);
}

void RISCVISAInfo::updateFLen() {
  FLen = 0;
  // TODO: Handle q extension.
  if (Exts.count("d"))
    FLen = 64;
  else if (Exts.count("f"))
    FLen = 32;
}

void RISCVISAInfo::updateMinVLen() {
  for (auto const &Ext : Exts) {
    StringRef ExtName = Ext.first;
    bool IsZvlExt = ExtName.consume_front("zvl") && ExtName.consume_back("b");
    if (IsZvlExt) {
      unsigned ZvlLen;
      if (!ExtName.getAsInteger(10, ZvlLen))
        MinVLen = std::max(MinVLen, ZvlLen);
    }
  }
}

void RISCVISAInfo::updateMaxELen() {
  // handles EEW restriction by sub-extension zve
  for (auto const &Ext : Exts) {
    StringRef ExtName = Ext.first;
    bool IsZveExt = ExtName.consume_front("zve");
    if (IsZveExt) {
      if (ExtName.back() == 'f')
        MaxELenFp = std::max(MaxELenFp, 32u);
      if (ExtName.back() == 'd')
        MaxELenFp = std::max(MaxELenFp, 64u);
      ExtName = ExtName.drop_back();
      unsigned ZveELen;
      ExtName.getAsInteger(10, ZveELen);
      MaxELen = std::max(MaxELen, ZveELen);
    }
  }
}

std::string RISCVISAInfo::toString() const {
  std::string Buffer;
  raw_string_ostream Arch(Buffer);

  Arch << "rv" << XLen;

  ListSeparator LS("_");
  for (auto const &Ext : Exts) {
    StringRef ExtName = Ext.first;
    auto ExtInfo = Ext.second;
    Arch << LS << ExtName;
    Arch << ExtInfo.Major << "p" << ExtInfo.Minor;
  }

  return Arch.str();
}

llvm::Expected<std::unique_ptr<RISCVISAInfo>>
RISCVISAInfo::postProcessAndChecking(std::unique_ptr<RISCVISAInfo> &&ISAInfo) {
  ISAInfo->updateImplication();
  ISAInfo->updateCombination();
  ISAInfo->updateFLen();
  ISAInfo->updateMinVLen();
  ISAInfo->updateMaxELen();

  if (Error Result = ISAInfo->checkDependency())
    return std::move(Result);
  return std::move(ISAInfo);
}

StringRef RISCVISAInfo::computeDefaultABI() const {
  if (XLen == 32) {
    if (hasExtension("e"))
      return "ilp32e";
    if (hasExtension("d"))
      return "ilp32d";
    if (hasExtension("f"))
      return "ilp32f";
    return "ilp32";
  } else if (XLen == 64) {
    if (hasExtension("e"))
      return "lp64e";
    if (hasExtension("d"))
      return "lp64d";
    if (hasExtension("f"))
      return "lp64f";
    return "lp64";
  }
  llvm_unreachable("Invalid XLEN");
}

bool RISCVISAInfo::isSupportedExtensionWithVersion(StringRef Ext) {
  if (Ext.empty())
    return false;

  auto Pos = findLastNonVersionCharacter(Ext) + 1;
  StringRef Name = Ext.substr(0, Pos);
  StringRef Vers = Ext.substr(Pos);
  if (Vers.empty())
    return false;

  unsigned Major, Minor, ConsumeLength;
  if (auto E = getExtensionVersion(Name, Vers, Major, Minor, ConsumeLength,
                                   true, true)) {
    consumeError(std::move(E));
    return false;
  }

  return true;
}

std::string RISCVISAInfo::getTargetFeatureForExtension(StringRef Ext) {
  if (Ext.empty())
    return std::string();

  auto Pos = findLastNonVersionCharacter(Ext) + 1;
  StringRef Name = Ext.substr(0, Pos);

  if (Pos != Ext.size() && !isSupportedExtensionWithVersion(Ext))
    return std::string();

  if (!isSupportedExtension(Name))
    return std::string();

  return isExperimentalExtension(Name) ? "experimental-" + Name.str()
                                       : Name.str();
}
