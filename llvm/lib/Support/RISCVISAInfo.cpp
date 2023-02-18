//===-- RISCVISAInfo.cpp - RISCV Arch String Parser -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;

namespace {
/// Represents the major and version number components of a RISC-V extension
struct RISCVExtensionVersion {
  unsigned Major;
  unsigned Minor;
};

struct RISCVSupportedExtension {
  const char *Name;
  /// Supported version.
  RISCVExtensionVersion Version;
};

} // end anonymous namespace

static constexpr StringLiteral AllStdExts = "mafdqlcbkjtpvnh";

static const RISCVSupportedExtension SupportedExtensions[] = {
    {"i", RISCVExtensionVersion{2, 0}},
    {"e", RISCVExtensionVersion{1, 9}},
    {"m", RISCVExtensionVersion{2, 0}},
    {"a", RISCVExtensionVersion{2, 0}},
    {"f", RISCVExtensionVersion{2, 0}},
    {"d", RISCVExtensionVersion{2, 0}},
    {"c", RISCVExtensionVersion{2, 0}},

    {"h", RISCVExtensionVersion{1, 0}},

    {"zihintpause", RISCVExtensionVersion{2, 0}},

    {"zfhmin", RISCVExtensionVersion{1, 0}},
    {"zfh", RISCVExtensionVersion{1, 0}},

    {"zfinx", RISCVExtensionVersion{1, 0}},
    {"zdinx", RISCVExtensionVersion{1, 0}},
    {"zhinxmin", RISCVExtensionVersion{1, 0}},
    {"zhinx", RISCVExtensionVersion{1, 0}},

    {"zba", RISCVExtensionVersion{1, 0}},
    {"zbb", RISCVExtensionVersion{1, 0}},
    {"zbc", RISCVExtensionVersion{1, 0}},
    {"zbs", RISCVExtensionVersion{1, 0}},

    {"zbkb", RISCVExtensionVersion{1, 0}},
    {"zbkc", RISCVExtensionVersion{1, 0}},
    {"zbkx", RISCVExtensionVersion{1, 0}},
    {"zknd", RISCVExtensionVersion{1, 0}},
    {"zkne", RISCVExtensionVersion{1, 0}},
    {"zknh", RISCVExtensionVersion{1, 0}},
    {"zksed", RISCVExtensionVersion{1, 0}},
    {"zksh", RISCVExtensionVersion{1, 0}},
    {"zkr", RISCVExtensionVersion{1, 0}},
    {"zkn", RISCVExtensionVersion{1, 0}},
    {"zks", RISCVExtensionVersion{1, 0}},
    {"zkt", RISCVExtensionVersion{1, 0}},
    {"zk", RISCVExtensionVersion{1, 0}},

    {"zmmul", RISCVExtensionVersion{1, 0}},

    {"v", RISCVExtensionVersion{1, 0}},
    {"zvl32b", RISCVExtensionVersion{1, 0}},
    {"zvl64b", RISCVExtensionVersion{1, 0}},
    {"zvl128b", RISCVExtensionVersion{1, 0}},
    {"zvl256b", RISCVExtensionVersion{1, 0}},
    {"zvl512b", RISCVExtensionVersion{1, 0}},
    {"zvl1024b", RISCVExtensionVersion{1, 0}},
    {"zvl2048b", RISCVExtensionVersion{1, 0}},
    {"zvl4096b", RISCVExtensionVersion{1, 0}},
    {"zvl8192b", RISCVExtensionVersion{1, 0}},
    {"zvl16384b", RISCVExtensionVersion{1, 0}},
    {"zvl32768b", RISCVExtensionVersion{1, 0}},
    {"zvl65536b", RISCVExtensionVersion{1, 0}},
    {"zve32x", RISCVExtensionVersion{1, 0}},
    {"zve32f", RISCVExtensionVersion{1, 0}},
    {"zve64x", RISCVExtensionVersion{1, 0}},
    {"zve64f", RISCVExtensionVersion{1, 0}},
    {"zve64d", RISCVExtensionVersion{1, 0}},

    {"zicbom", RISCVExtensionVersion{1, 0}},
    {"zicboz", RISCVExtensionVersion{1, 0}},
    {"zicbop", RISCVExtensionVersion{1, 0}},
    {"zicsr", RISCVExtensionVersion{2, 0}},
    {"zifencei", RISCVExtensionVersion{2, 0}},

    {"svnapot", RISCVExtensionVersion{1, 0}},
    {"svpbmt", RISCVExtensionVersion{1, 0}},
    {"svinval", RISCVExtensionVersion{1, 0}},

    // vendor-defined ('X') extensions
    {"xtheadba", RISCVExtensionVersion{1, 0}},
    {"xtheadbb", RISCVExtensionVersion{1, 0}},
    {"xtheadbs", RISCVExtensionVersion{1, 0}},
    {"xtheadmac", RISCVExtensionVersion{1, 0}},
    {"xtheadvdot", RISCVExtensionVersion{1, 0}},
    {"xventanacondops", RISCVExtensionVersion{1, 0}},
};

static const RISCVSupportedExtension SupportedExperimentalExtensions[] = {
    {"zihintntl", RISCVExtensionVersion{0, 2}},

    {"zca", RISCVExtensionVersion{1, 0}},
    {"zcb", RISCVExtensionVersion{1, 0}},
    {"zcd", RISCVExtensionVersion{1, 0}},
    {"zcf", RISCVExtensionVersion{1, 0}},
    {"zfa", RISCVExtensionVersion{0, 1}},
    {"zvfh", RISCVExtensionVersion{0, 1}},
    {"zawrs", RISCVExtensionVersion{1, 0}},
    {"ztso", RISCVExtensionVersion{0, 1}},
};

static bool stripExperimentalPrefix(StringRef &Ext) {
  return Ext.consume_front("experimental-");
}

// This function finds the first character that doesn't belong to a version
// (e.g. zba1p0 is extension 'zba' of version '1p0'). So the function will
// consume [0-9]*p[0-9]* starting from the backward. An extension name will not
// end with a digit or the letter 'p', so this function will parse correctly.
// NOTE: This function is NOT able to take empty strings or strings that only
// have version numbers and no extension name. It assumes the extension name
// will be at least more than one character.
static size_t findFirstNonVersionCharacter(StringRef Ext) {
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
struct FindByName {
  FindByName(StringRef Ext) : Ext(Ext){};
  StringRef Ext;
  bool operator()(const RISCVSupportedExtension &ExtInfo) {
    return ExtInfo.Name == Ext;
  }
};
} // namespace

static std::optional<RISCVExtensionVersion>
findDefaultVersion(StringRef ExtName) {
  // Find default version of an extension.
  // TODO: We might set default version based on profile or ISA spec.
  for (auto &ExtInfo : {ArrayRef(SupportedExtensions),
                        ArrayRef(SupportedExperimentalExtensions)}) {
    auto ExtensionInfoIterator = llvm::find_if(ExtInfo, FindByName(ExtName));

    if (ExtensionInfoIterator == ExtInfo.end()) {
      continue;
    }
    return ExtensionInfoIterator->Version;
  }
  return std::nullopt;
}

void RISCVISAInfo::addExtension(StringRef ExtName, unsigned MajorVersion,
                                unsigned MinorVersion) {
  RISCVExtensionInfo Ext;
  Ext.ExtName = ExtName.str();
  Ext.MajorVersion = MajorVersion;
  Ext.MinorVersion = MinorVersion;
  Exts[ExtName.str()] = Ext;
}

static StringRef getExtensionTypeDesc(StringRef Ext) {
  if (Ext.startswith("sx"))
    return "non-standard supervisor-level extension";
  if (Ext.startswith("s"))
    return "standard supervisor-level extension";
  if (Ext.startswith("x"))
    return "non-standard user-level extension";
  if (Ext.startswith("z"))
    return "standard user-level extension";
  return StringRef();
}

static StringRef getExtensionType(StringRef Ext) {
  if (Ext.startswith("sx"))
    return "sx";
  if (Ext.startswith("s"))
    return "s";
  if (Ext.startswith("x"))
    return "x";
  if (Ext.startswith("z"))
    return "z";
  return StringRef();
}

static std::optional<RISCVExtensionVersion>
isExperimentalExtension(StringRef Ext) {
  auto ExtIterator =
      llvm::find_if(SupportedExperimentalExtensions, FindByName(Ext));
  if (ExtIterator == std::end(SupportedExperimentalExtensions))
    return std::nullopt;

  return ExtIterator->Version;
}

bool RISCVISAInfo::isSupportedExtensionFeature(StringRef Ext) {
  bool IsExperimental = stripExperimentalPrefix(Ext);

  if (IsExperimental)
    return llvm::any_of(SupportedExperimentalExtensions, FindByName(Ext));
  else
    return llvm::any_of(SupportedExtensions, FindByName(Ext));
}

bool RISCVISAInfo::isSupportedExtension(StringRef Ext) {
  return llvm::any_of(SupportedExtensions, FindByName(Ext)) ||
         llvm::any_of(SupportedExperimentalExtensions, FindByName(Ext));
}

bool RISCVISAInfo::isSupportedExtension(StringRef Ext, unsigned MajorVersion,
                                        unsigned MinorVersion) {
  auto FindByNameAndVersion = [=](const RISCVSupportedExtension &ExtInfo) {
    return ExtInfo.Name == Ext && (MajorVersion == ExtInfo.Version.Major) &&
           (MinorVersion == ExtInfo.Version.Minor);
  };
  return llvm::any_of(SupportedExtensions, FindByNameAndVersion) ||
         llvm::any_of(SupportedExperimentalExtensions, FindByNameAndVersion);
}

bool RISCVISAInfo::hasExtension(StringRef Ext) const {
  stripExperimentalPrefix(Ext);

  if (!isSupportedExtension(Ext))
    return false;

  return Exts.count(Ext.str()) != 0;
}

// Get the rank for single-letter extension, lower value meaning higher
// priority.
static int singleLetterExtensionRank(char Ext) {
  switch (Ext) {
  case 'i':
    return -2;
  case 'e':
    return -1;
  default:
    break;
  }

  size_t Pos = AllStdExts.find(Ext);
  int Rank;
  if (Pos == StringRef::npos)
    // If we got an unknown extension letter, then give it an alphabetical
    // order, but after all known standard extensions.
    Rank = AllStdExts.size() + (Ext - 'a');
  else
    Rank = Pos;

  return Rank;
}

// Get the rank for multi-letter extension, lower value meaning higher
// priority/order in canonical order.
static int multiLetterExtensionRank(const std::string &ExtName) {
  assert(ExtName.length() >= 2);
  int HighOrder;
  int LowOrder = 0;
  // The order between multi-char extensions: s -> h -> z -> x.
  char ExtClass = ExtName[0];
  switch (ExtClass) {
  case 's':
    HighOrder = 0;
    break;
  case 'z':
    HighOrder = 1;
    // `z` extension must be sorted by canonical order of second letter.
    // e.g. zmx has higher rank than zax.
    LowOrder = singleLetterExtensionRank(ExtName[1]);
    break;
  case 'x':
    HighOrder = 2;
    break;
  default:
    llvm_unreachable("Unknown prefix for multi-char extension");
    return -1;
  }

  return (HighOrder << 8) + LowOrder;
}

// Compare function for extension.
// Only compare the extension name, ignore version comparison.
bool RISCVISAInfo::compareExtension(const std::string &LHS,
                                    const std::string &RHS) {
  size_t LHSLen = LHS.length();
  size_t RHSLen = RHS.length();
  if (LHSLen == 1 && RHSLen != 1)
    return true;

  if (LHSLen != 1 && RHSLen == 1)
    return false;

  if (LHSLen == 1 && RHSLen == 1)
    return singleLetterExtensionRank(LHS[0]) <
           singleLetterExtensionRank(RHS[0]);

  // Both are multi-char ext here.
  int LHSRank = multiLetterExtensionRank(LHS);
  int RHSRank = multiLetterExtensionRank(RHS);
  if (LHSRank != RHSRank)
    return LHSRank < RHSRank;

  // If the rank is same, it must be sorted by lexicographic order.
  return LHS < RHS;
}

void RISCVISAInfo::toFeatures(
    std::vector<StringRef> &Features,
    llvm::function_ref<StringRef(const Twine &)> StrAlloc,
    bool AddAllExtensions) const {
  for (auto const &Ext : Exts) {
    StringRef ExtName = Ext.first;

    if (ExtName == "i")
      continue;

    if (isExperimentalExtension(ExtName)) {
      Features.push_back(StrAlloc("+experimental-" + ExtName));
    } else {
      Features.push_back(StrAlloc("+" + ExtName));
    }
  }
  if (AddAllExtensions) {
    for (const RISCVSupportedExtension &Ext : SupportedExtensions) {
      if (Exts.count(Ext.Name))
        continue;
      Features.push_back(StrAlloc(Twine("-") + Ext.Name));
    }

    for (const RISCVSupportedExtension &Ext : SupportedExperimentalExtensions) {
      if (Exts.count(Ext.Name))
        continue;
      Features.push_back(StrAlloc(Twine("-experimental-") + Ext.Name));
    }
  }
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
  if (Ext.size() > 1 && In.size()) {
    std::string Error =
        "multi-character extensions must be separated by underscores";
    return createStringError(errc::invalid_argument, Error);
  }

  // If experimental extension, require use of current version number number
  if (auto ExperimentalExtension = isExperimentalExtension(Ext)) {
    if (!EnableExperimentalExtension) {
      std::string Error = "requires '-menable-experimental-extensions' for "
                          "experimental extension '" +
                          Ext.str() + "'";
      return createStringError(errc::invalid_argument, Error);
    }

    if (ExperimentalExtensionVersionCheck &&
        (MajorStr.empty() && MinorStr.empty())) {
      std::string Error =
          "experimental extension requires explicit version number `" +
          Ext.str() + "`";
      return createStringError(errc::invalid_argument, Error);
    }

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
        llvm::find_if(ExtensionInfos, FindByName(ExtName));

    // Not all features is related to ISA extension, like `relax` or
    // `save-restore`, skip those feature.
    if (ExtensionInfoIterator == ExtensionInfos.end())
      continue;

    if (Add)
      ISAInfo->addExtension(ExtName, ExtensionInfoIterator->Version.Major,
                            ExtensionInfoIterator->Version.Minor);
    else
      ISAInfo->Exts.erase(ExtName.str());
  }

  return RISCVISAInfo::postProcessAndChecking(std::move(ISAInfo));
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

  bool HasRV64 = Arch.startswith("rv64");
  // ISA string must begin with rv32 or rv64.
  if (!(Arch.startswith("rv32") || HasRV64) || (Arch.size() < 5)) {
    return createStringError(errc::invalid_argument,
                             "string must begin with rv32{i,e,g} or rv64{i,g}");
  }

  unsigned XLen = HasRV64 ? 64 : 32;
  std::unique_ptr<RISCVISAInfo> ISAInfo(new RISCVISAInfo(XLen));

  // The canonical order specified in ISA manual.
  // Ref: Table 22.1 in RISC-V User-Level ISA V2.2
  StringRef StdExts = AllStdExts;
  char Baseline = Arch[4];

  // First letter should be 'e', 'i' or 'g'.
  switch (Baseline) {
  default:
    return createStringError(errc::invalid_argument,
                             "first letter should be 'e', 'i' or 'g'");
  case 'e': {
    // Extension 'e' is not allowed in rv64.
    if (HasRV64)
      return createStringError(
          errc::invalid_argument,
          "standard user-level extension 'e' requires 'rv32'");
    break;
  }
  case 'i':
    break;
  case 'g':
    // g = imafd
    StdExts = StdExts.drop_front(4);
    break;
  }

  // Skip rvxxx
  StringRef Exts = Arch.substr(5);

  // Remove multi-letter standard extensions, non-standard extensions and
  // supervisor-level extensions. They have 'z', 'x', 's', 'sx' prefixes.
  // Parse them at the end.
  // Find the very first occurrence of 's', 'x' or 'z'.
  StringRef OtherExts;
  size_t Pos = Exts.find_first_of("zsx");
  if (Pos != StringRef::npos) {
    OtherExts = Exts.substr(Pos);
    Exts = Exts.substr(0, Pos);
  }

  unsigned Major, Minor, ConsumeLength;
  if (auto E = getExtensionVersion(std::string(1, Baseline), Exts, Major, Minor,
                                   ConsumeLength, EnableExperimentalExtension,
                                   ExperimentalExtensionVersionCheck))
    return std::move(E);

  if (Baseline == 'g') {
    // No matter which version is given to `g`, we always set imafd to default
    // version since the we don't have clear version scheme for that on
    // ISA spec.
    for (const auto *Ext : {"i", "m", "a", "f", "d"})
      if (auto Version = findDefaultVersion(Ext))
        ISAInfo->addExtension(Ext, Version->Major, Version->Minor);
      else
        llvm_unreachable("Default extension version not found?");
  } else
    // Baseline is `i` or `e`
    ISAInfo->addExtension(std::string(1, Baseline), Major, Minor);

  // Consume the base ISA version number and any '_' between rvxxx and the
  // first extension
  Exts = Exts.drop_front(ConsumeLength);
  Exts.consume_front("_");

  // TODO: Use version number when setting target features

  auto StdExtsItr = StdExts.begin();
  auto StdExtsEnd = StdExts.end();
  auto GoToNextExt = [](StringRef::iterator &I, unsigned ConsumeLength) {
    I += 1 + ConsumeLength;
    if (*I == '_')
      ++I;
  };
  for (auto I = Exts.begin(), E = Exts.end(); I != E;) {
    char C = *I;

    // Check ISA extensions are specified in the canonical order.
    while (StdExtsItr != StdExtsEnd && *StdExtsItr != C)
      ++StdExtsItr;

    if (StdExtsItr == StdExtsEnd) {
      // Either c contains a valid extension but it was not given in
      // canonical order or it is an invalid extension.
      if (StdExts.contains(C)) {
        return createStringError(
            errc::invalid_argument,
            "standard user-level extension not given in canonical order '%c'",
            C);
      }

      return createStringError(errc::invalid_argument,
                               "invalid standard user-level extension '%c'", C);
    }

    // Move to next char to prevent repeated letter.
    ++StdExtsItr;

    std::string Next;
    unsigned Major, Minor, ConsumeLength;
    if (std::next(I) != E)
      Next = std::string(std::next(I), E);
    if (auto E = getExtensionVersion(std::string(1, C), Next, Major, Minor,
                                     ConsumeLength, EnableExperimentalExtension,
                                     ExperimentalExtensionVersionCheck)) {
      if (IgnoreUnknown) {
        consumeError(std::move(E));
        GoToNextExt(I, ConsumeLength);
        continue;
      }
      return std::move(E);
    }

    // The order is OK, then push it into features.
    // TODO: Use version number when setting target features
    // Currently LLVM supports only "mafdcvh".
    if (!isSupportedExtension(StringRef(&C, 1))) {
      if (IgnoreUnknown) {
        GoToNextExt(I, ConsumeLength);
        continue;
      }
      return createStringError(errc::invalid_argument,
                               "unsupported standard user-level extension '%c'",
                               C);
    }
    ISAInfo->addExtension(std::string(1, C), Major, Minor);

    // Consume full extension name and version, including any optional '_'
    // between this extension and the next
    GoToNextExt(I, ConsumeLength);
  }

  // Handle other types of extensions other than the standard
  // general purpose and standard user-level extensions.
  // Parse the ISA string containing non-standard user-level
  // extensions, standard supervisor-level extensions and
  // non-standard supervisor-level extensions.
  // These extensions start with 'z', 'x', 's', 'sx' prefixes, follow a
  // canonical order, might have a version number (major, minor)
  // and are separated by a single underscore '_'.
  // Set the hardware features for the extensions that are supported.

  // Multi-letter extensions are seperated by a single underscore
  // as described in RISC-V User-Level ISA V2.2.
  SmallVector<StringRef, 8> Split;
  OtherExts.split(Split, '_');

  SmallVector<StringRef, 8> AllExts;
  std::array<StringRef, 4> Prefix{"z", "x", "s", "sx"};
  auto I = Prefix.begin();
  auto E = Prefix.end();
  if (Split.size() > 1 || Split[0] != "") {
    for (StringRef Ext : Split) {
      if (Ext.empty())
        return createStringError(errc::invalid_argument,
                                 "extension name missing after separator '_'");

      StringRef Type = getExtensionType(Ext);
      StringRef Desc = getExtensionTypeDesc(Ext);
      auto Pos = findFirstNonVersionCharacter(Ext) + 1;
      StringRef Name(Ext.substr(0, Pos));
      StringRef Vers(Ext.substr(Pos));

      if (Type.empty()) {
        if (IgnoreUnknown)
          continue;
        return createStringError(errc::invalid_argument,
                                 "invalid extension prefix '" + Ext + "'");
      }

      // Check ISA extensions are specified in the canonical order.
      while (I != E && *I != Type)
        ++I;

      if (I == E) {
        if (IgnoreUnknown)
          continue;
        return createStringError(errc::invalid_argument,
                                 "%s not given in canonical order '%s'",
                                 Desc.str().c_str(), Ext.str().c_str());
      }

      if (!IgnoreUnknown && Name.size() == Type.size()) {
        return createStringError(errc::invalid_argument,
                                 "%s name missing after '%s'",
                                 Desc.str().c_str(), Type.str().c_str());
      }

      unsigned Major, Minor, ConsumeLength;
      if (auto E = getExtensionVersion(Name, Vers, Major, Minor, ConsumeLength,
                                       EnableExperimentalExtension,
                                       ExperimentalExtensionVersionCheck)) {
        if (IgnoreUnknown) {
          consumeError(std::move(E));
          continue;
        }
        return std::move(E);
      }

      // Check if duplicated extension.
      if (!IgnoreUnknown && llvm::is_contained(AllExts, Name)) {
        return createStringError(errc::invalid_argument, "duplicated %s '%s'",
                                 Desc.str().c_str(), Name.str().c_str());
      }

      ISAInfo->addExtension(Name, Major, Minor);
      // Extension format is correct, keep parsing the extensions.
      // TODO: Save Type, Name, Major, Minor to avoid parsing them later.
      AllExts.push_back(Name);
    }
  }

  for (auto Ext : AllExts) {
    if (!isSupportedExtension(Ext)) {
      StringRef Desc = getExtensionTypeDesc(getExtensionType(Ext));
      return createStringError(errc::invalid_argument, "unsupported %s '%s'",
                               Desc.str().c_str(), Ext.str().c_str());
    }
  }

  return RISCVISAInfo::postProcessAndChecking(std::move(ISAInfo));
}

Error RISCVISAInfo::checkDependency() {
  bool IsRv32 = XLen == 32;
  bool HasE = Exts.count("e") != 0;
  bool HasD = Exts.count("d") != 0;
  bool HasF = Exts.count("f") != 0;
  bool HasZfinx = Exts.count("zfinx") != 0;
  bool HasZdinx = Exts.count("zdinx") != 0;
  bool HasVector = Exts.count("zve32x") != 0;
  bool HasZve32f = Exts.count("zve32f") != 0;
  bool HasZve64d = Exts.count("zve64d") != 0;
  bool HasZvl = MinVLen != 0;

  if (HasE && !IsRv32)
    return createStringError(
        errc::invalid_argument,
        "standard user-level extension 'e' requires 'rv32'");

  // It's illegal to specify the 'd' (double-precision floating point)
  // extension without also specifying the 'f' (single precision
  // floating-point) extension.
  // TODO: This has been removed in later specs, which specify that D implies F
  if (HasD && !HasF)
    return createStringError(errc::invalid_argument,
                             "d requires f extension to also be specified");

  if (HasZve32f && !HasF && !HasZfinx)
    return createStringError(
        errc::invalid_argument,
        "zve32f requires f or zfinx extension to also be specified");

  if (HasZve64d && !HasD && !HasZdinx)
    return createStringError(
        errc::invalid_argument,
        "zve64d requires d or zdinx extension to also be specified");

  if (Exts.count("zvfh") && !Exts.count("zfh") && !Exts.count("zfhmin") &&
      !Exts.count("zhinx") && !Exts.count("zhinxmin"))
    return createStringError(
        errc::invalid_argument,
        "zvfh requires zfh, zfhmin, zhinx or zhinxmin extension to also be "
        "specified");

  if (HasZvl && !HasVector)
    return createStringError(
        errc::invalid_argument,
        "zvl*b requires v or zve* extension to also be specified");

  // Additional dependency checks.
  // TODO: The 'q' extension requires rv64.
  // TODO: It is illegal to specify 'e' extensions with 'f' and 'd'.

  return Error::success();
}

static const char *ImpliedExtsV[] = {"zvl128b", "zve64d", "f", "d"};
static const char *ImpliedExtsZfhmin[] = {"f"};
static const char *ImpliedExtsZfh[] = {"f"};
static const char *ImpliedExtsZdinx[] = {"zfinx"};
static const char *ImpliedExtsZhinxmin[] = {"zfinx"};
static const char *ImpliedExtsZhinx[] = {"zfinx"};
static const char *ImpliedExtsZve64d[] = {"zve64f"};
static const char *ImpliedExtsZve64f[] = {"zve64x", "zve32f"};
static const char *ImpliedExtsZve64x[] = {"zve32x", "zvl64b"};
static const char *ImpliedExtsZve32f[] = {"zve32x"};
static const char *ImpliedExtsZve32x[] = {"zvl32b"};
static const char *ImpliedExtsZvl65536b[] = {"zvl32768b"};
static const char *ImpliedExtsZvl32768b[] = {"zvl16384b"};
static const char *ImpliedExtsZvl16384b[] = {"zvl8192b"};
static const char *ImpliedExtsZvl8192b[] = {"zvl4096b"};
static const char *ImpliedExtsZvl4096b[] = {"zvl2048b"};
static const char *ImpliedExtsZvl2048b[] = {"zvl1024b"};
static const char *ImpliedExtsZvl1024b[] = {"zvl512b"};
static const char *ImpliedExtsZvl512b[] = {"zvl256b"};
static const char *ImpliedExtsZvl256b[] = {"zvl128b"};
static const char *ImpliedExtsZvl128b[] = {"zvl64b"};
static const char *ImpliedExtsZvl64b[] = {"zvl32b"};
static const char *ImpliedExtsZk[] = {"zkn", "zkt", "zkr"};
static const char *ImpliedExtsZkn[] = {"zbkb", "zbkc", "zbkx",
                                       "zkne", "zknd", "zknh"};
static const char *ImpliedExtsZks[] = {"zbkb", "zbkc", "zbkx", "zksed", "zksh"};
static const char *ImpliedExtsZvfh[] = {"zve32f"};
static const char *ImpliedExtsXTHeadVdot[] = {"v"};
static const char *ImpliedExtsZcb[] = {"zca"};
static const char *ImpliedExtsZfa[] = {"f"};

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
    {{"v"}, {ImpliedExtsV}},
    {{"xtheadvdot"}, {ImpliedExtsXTHeadVdot}},
    {{"zcb"}, {ImpliedExtsZcb}},
    {{"zdinx"}, {ImpliedExtsZdinx}},
    {{"zfa"}, {ImpliedExtsZfa}},
    {{"zfh"}, {ImpliedExtsZfh}},
    {{"zfhmin"}, {ImpliedExtsZfhmin}},
    {{"zhinx"}, {ImpliedExtsZhinx}},
    {{"zhinxmin"}, {ImpliedExtsZhinxmin}},
    {{"zk"}, {ImpliedExtsZk}},
    {{"zkn"}, {ImpliedExtsZkn}},
    {{"zks"}, {ImpliedExtsZks}},
    {{"zve32f"}, {ImpliedExtsZve32f}},
    {{"zve32x"}, {ImpliedExtsZve32x}},
    {{"zve64d"}, {ImpliedExtsZve64d}},
    {{"zve64f"}, {ImpliedExtsZve64f}},
    {{"zve64x"}, {ImpliedExtsZve64x}},
    {{"zvfh"}, {ImpliedExtsZvfh}},
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
    addExtension("i", Version->Major, Version->Minor);
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
        addExtension(ImpliedExt, Version->Major, Version->Minor);
        WorkList.insert(ImpliedExt);
      }
    }
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
        addExtension(CombineExt, Version->Major, Version->Minor);
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
    Arch << ExtInfo.MajorVersion << "p" << ExtInfo.MinorVersion;
  }

  return Arch.str();
}

std::vector<std::string> RISCVISAInfo::toFeatureVector() const {
  std::vector<std::string> FeatureVector;
  for (auto const &Ext : Exts) {
    std::string ExtName = Ext.first;
    if (ExtName == "i") // i is not recognized in clang -cc1
      continue;
    std::string Feature = isExperimentalExtension(ExtName)
                              ? "+experimental-" + ExtName
                              : "+" + ExtName;
    FeatureVector.push_back(Feature);
  }
  return FeatureVector;
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
    if (hasExtension("d"))
      return "ilp32d";
    if (hasExtension("e"))
      return "ilp32e";
    return "ilp32";
  } else if (XLen == 64) {
    if (hasExtension("d"))
      return "lp64d";
    return "lp64";
  }
  llvm_unreachable("Invalid XLEN");
}
