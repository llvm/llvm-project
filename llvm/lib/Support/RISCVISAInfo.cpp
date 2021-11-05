//===-- RISCVISAInfo.cpp - RISCV Arch String Parser --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
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

static constexpr StringLiteral AllStdExts = "mafdqlcbjtpvn";

static const RISCVSupportedExtension SupportedExtensions[] = {
    {"i", RISCVExtensionVersion{2, 0}},
    {"e", RISCVExtensionVersion{1, 9}},
    {"m", RISCVExtensionVersion{2, 0}},
    {"a", RISCVExtensionVersion{2, 0}},
    {"f", RISCVExtensionVersion{2, 0}},
    {"d", RISCVExtensionVersion{2, 0}},
    {"c", RISCVExtensionVersion{2, 0}},
};

static const RISCVSupportedExtension SupportedExperimentalExtensions[] = {
    {"v", RISCVExtensionVersion{0, 10}},
    {"zba", RISCVExtensionVersion{1, 0}},
    {"zbb", RISCVExtensionVersion{1, 0}},
    {"zbc", RISCVExtensionVersion{1, 0}},
    {"zbe", RISCVExtensionVersion{0, 93}},
    {"zbf", RISCVExtensionVersion{0, 93}},
    {"zbm", RISCVExtensionVersion{0, 93}},
    {"zbp", RISCVExtensionVersion{0, 93}},
    {"zbr", RISCVExtensionVersion{0, 93}},
    {"zbs", RISCVExtensionVersion{1, 0}},
    {"zbt", RISCVExtensionVersion{0, 93}},

    {"zvamo", RISCVExtensionVersion{0, 10}},
    {"zvlsseg", RISCVExtensionVersion{0, 10}},

    {"zfhmin", RISCVExtensionVersion{0, 1}},
    {"zfh", RISCVExtensionVersion{0, 1}},
};

static bool stripExperimentalPrefix(StringRef &Ext) {
  return Ext.consume_front("experimental-");
}

struct FindByName {
  FindByName(StringRef Ext) : Ext(Ext){};
  StringRef Ext;
  bool operator()(const RISCVSupportedExtension &ExtInfo) {
    return ExtInfo.Name == Ext;
  }
};

static Optional<RISCVExtensionVersion> findDefaultVersion(StringRef ExtName) {
  // Find default version of an extension.
  // TODO: We might set default version based on profile or ISA spec.
  for (auto &ExtInfo : {makeArrayRef(SupportedExtensions),
                        makeArrayRef(SupportedExperimentalExtensions)}) {
    auto ExtensionInfoIterator = llvm::find_if(ExtInfo, FindByName(ExtName));

    if (ExtensionInfoIterator == ExtInfo.end()) {
      continue;
    }
    return ExtensionInfoIterator->Version;
  }
  return None;
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

static Optional<RISCVExtensionVersion> isExperimentalExtension(StringRef Ext) {
  auto ExtIterator =
      llvm::find_if(SupportedExperimentalExtensions, FindByName(Ext));
  if (ExtIterator == std::end(SupportedExperimentalExtensions))
    return None;

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
  case 'h':
    HighOrder = 1;
    break;
  case 'z':
    HighOrder = 2;
    // `z` extension must be sorted by canonical order of second letter.
    // e.g. zmx has higher rank than zax.
    LowOrder = singleLetterExtensionRank(ExtName[1]);
    break;
  case 'x':
    HighOrder = 3;
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
    std::function<StringRef(const Twine &)> StrAlloc) const {
  for (auto &Ext : Exts) {
    StringRef ExtName = Ext.first;

    if (ExtName == "i")
      continue;

    if (ExtName == "zvlsseg") {
      Features.push_back("+experimental-v");
      Features.push_back("+experimental-zvlsseg");
    } else if (ExtName == "zvamo") {
      Features.push_back("+experimental-v");
      Features.push_back("+experimental-zvlsseg");
      Features.push_back("+experimental-zvamo");
    } else if (isExperimentalExtension(ExtName)) {
      Features.push_back(StrAlloc("+experimental-" + ExtName));
    } else {
      Features.push_back(StrAlloc("+" + ExtName));
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
    In = In.substr(MajorStr.size() + 1);

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
               "'(this compiler supports " + utostr(SupportedVers.Major) + "." +
               utostr(SupportedVers.Minor) + ")";
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

  bool HasE = false;
  for (auto &Feature : Features) {
    StringRef ExtName = Feature;
    bool Experimental = false;
    assert(ExtName.size() > 1 && (ExtName[0] == '+' || ExtName[0] == '-'));
    bool Add = ExtName[0] == '+';
    ExtName = ExtName.drop_front(1); // Drop '+' or '-'
    Experimental = stripExperimentalPrefix(ExtName);
    auto ExtensionInfos = Experimental
                              ? makeArrayRef(SupportedExperimentalExtensions)
                              : makeArrayRef(SupportedExtensions);
    auto ExtensionInfoIterator =
        llvm::find_if(ExtensionInfos, FindByName(ExtName));

    // Not all features is related to ISA extension, like `relax` or
    // `save-restore`, skip those feature.
    if (ExtensionInfoIterator == ExtensionInfos.end())
      continue;

    if (Add) {
      if (ExtName == "e") {
        if (XLen != 32)
          return createStringError(
              errc::invalid_argument,
              "standard user-level extension 'e' requires 'rv32'");
        HasE = true;
      }

      ISAInfo->addExtension(ExtName, ExtensionInfoIterator->Version.Major,
                            ExtensionInfoIterator->Version.Minor);
    } else
      ISAInfo->Exts.erase(ExtName.str());
  }
  if (!HasE) {
    if (auto Version = findDefaultVersion("i"))
      ISAInfo->addExtension("i", Version->Major, Version->Minor);
    else
      llvm_unreachable("Default extension version for 'i' not found?");
  }

  ISAInfo->updateFLen();

  return std::move(ISAInfo);
}

llvm::Expected<std::unique_ptr<RISCVISAInfo>>
RISCVISAInfo::parseArchString(StringRef Arch, bool EnableExperimentalExtension,
                              bool ExperimentalExtensionVersionCheck) {
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
  bool HasF = false, HasD = false;
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
    HasF = true;
    HasD = true;
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
    for (auto Ext : {"i", "m", "a", "f", "d"})
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
                                     ExperimentalExtensionVersionCheck))
      return std::move(E);

    // The order is OK, then push it into features.
    // TODO: Use version number when setting target features
    switch (C) {
    default:
      // Currently LLVM supports only "mafdcbv".
      return createStringError(errc::invalid_argument,
                               "unsupported standard user-level extension '%c'",
                               C);
    case 'm':
      ISAInfo->addExtension("m", Major, Minor);
      break;
    case 'a':
      ISAInfo->addExtension("a", Major, Minor);
      break;
    case 'f':
      ISAInfo->addExtension("f", Major, Minor);
      HasF = true;
      break;
    case 'd':
      ISAInfo->addExtension("d", Major, Minor);
      HasD = true;
      break;
    case 'c':
      ISAInfo->addExtension("c", Major, Minor);
      break;
    case 'v':
      ISAInfo->addExtension("v", Major, Minor);
      ISAInfo->addExtension("zvlsseg", Major, Minor);
      break;
    }
    // Consume full extension name and version, including any optional '_'
    // between this extension and the next
    ++I;
    I += ConsumeLength;
    if (*I == '_')
      ++I;
  }
  // Dependency check.
  // It's illegal to specify the 'd' (double-precision floating point)
  // extension without also specifying the 'f' (single precision
  // floating-point) extension.
  // TODO: This has been removed in later specs, which specify that D implies F
  if (HasD && !HasF)
    return createStringError(errc::invalid_argument,
                             "d requires f extension to also be specified");

  // Additional dependency checks.
  // TODO: The 'q' extension requires rv64.
  // TODO: It is illegal to specify 'e' extensions with 'f' and 'd'.

  if (OtherExts.empty())
    return std::move(ISAInfo);

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

  for (StringRef Ext : Split) {
    if (Ext.empty())
      return createStringError(errc::invalid_argument,
                               "extension name missing after separator '_'");

    StringRef Type = getExtensionType(Ext);
    StringRef Desc = getExtensionTypeDesc(Ext);
    auto Pos = Ext.find_if(isDigit);
    StringRef Name(Ext.substr(0, Pos));
    StringRef Vers(Ext.substr(Pos));

    if (Type.empty())
      return createStringError(errc::invalid_argument,
                               "invalid extension prefix '" + Ext + "'");

    // Check ISA extensions are specified in the canonical order.
    while (I != E && *I != Type)
      ++I;

    if (I == E)
      return createStringError(errc::invalid_argument,
                               "%s not given in canonical order '%s'",
                               Desc.str().c_str(), Ext.str().c_str());

    if (Name.size() == Type.size()) {
      return createStringError(errc::invalid_argument,
                               "%s name missing after '%s'", Desc.str().c_str(),
                               Type.str().c_str());
    }

    unsigned Major, Minor, ConsumeLength;
    if (auto E = getExtensionVersion(Name, Vers, Major, Minor, ConsumeLength,
                                     EnableExperimentalExtension,
                                     ExperimentalExtensionVersionCheck))
      return std::move(E);

    // Check if duplicated extension.
    if (llvm::is_contained(AllExts, Name))
      return createStringError(errc::invalid_argument, "duplicated %s '%s'",
                               Desc.str().c_str(), Name.str().c_str());

    ISAInfo->addExtension(Name, Major, Minor);
    // Extension format is correct, keep parsing the extensions.
    // TODO: Save Type, Name, Major, Minor to avoid parsing them later.
    AllExts.push_back(Name);
  }

  for (auto Ext : AllExts) {
    if (!isSupportedExtension(Ext)) {
      StringRef Desc = getExtensionTypeDesc(getExtensionType(Ext));
      return createStringError(errc::invalid_argument, "unsupported %s '%s'",
                               Desc.str().c_str(), Ext.str().c_str());
    }
  }

  ISAInfo->updateFLen();

  return std::move(ISAInfo);
}

void RISCVISAInfo::updateFLen() {
  FLen = 0;
  // TODO: Handle q extension.
  if (Exts.count("d"))
    FLen = 64;
  else if (Exts.count("f"))
    FLen = 32;
}

std::string RISCVISAInfo::toString() const {
  std::string Buffer;
  raw_string_ostream Arch(Buffer);

  Arch << "rv" << XLen;

  ListSeparator LS("_");
  for (auto &Ext : Exts) {
    StringRef ExtName = Ext.first;
    auto ExtInfo = Ext.second;
    Arch << LS << ExtName;
    Arch << ExtInfo.MajorVersion << "p" << ExtInfo.MinorVersion;
  }

  return Arch.str();
}
