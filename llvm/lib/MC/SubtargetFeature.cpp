//===- SubtargetFeature.cpp - CPU characteristics Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SubtargetFeature interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                          Static Helper Functions
//===----------------------------------------------------------------------===//

/// hasFlag - Determine if a feature has a flag; '+' or '-'
///
static inline bool hasFlag(const StringRef Feature) {
  assert(!Feature.empty() && "Empty string");
  // Get first character
  char Ch = Feature[0];
  // Check if first character is '+' or '-' flag
  return Ch == '+' || Ch =='-';
}

/// StripFlag - Return string stripped of flag.
///
static inline std::string StripFlag(const StringRef Feature) {
  return hasFlag(Feature) ? Feature.substr(1) : Feature;
}

/// isEnabled - Return true if enable flag; '+'.
///
static inline bool isEnabled(const StringRef Feature) {
  assert(!Feature.empty() && "Empty string");
  // Get first character
  char Ch = Feature[0];
  // Check if first character is '+' for enabled
  return Ch == '+';
}

/// PrependFlag - Return a string with a prepended flag; '+' or '-'.
///
static inline std::string PrependFlag(const StringRef Feature,
                                    bool IsEnabled) {
  assert(!Feature.empty() && "Empty string");
  if (hasFlag(Feature))
    return Feature;
  std::string Prefix = IsEnabled ? "+" : "-";
  Prefix += Feature;
  return Prefix;
}

/// Split - Splits a string of comma separated items in to a vector of strings.
///
static void Split(std::vector<std::string> &V, const StringRef S) {
  if (S.empty())
    return;

  // Start at beginning of string.
  size_t Pos = 0;
  while (true) {
    // Find the next comma
    size_t Comma = S.find(',', Pos);
    // If no comma found then the rest of the string is used
    if (Comma == std::string::npos) {
      // Add string to vector
      V.push_back(S.substr(Pos));
      break;
    }
    // Otherwise add substring to vector
    V.push_back(S.substr(Pos, Comma - Pos));
    // Advance to next item
    Pos = Comma + 1;
  }
}

/// Join a vector of strings to a string with a comma separating each element.
///
static std::string Join(const std::vector<std::string> &V) {
  // Start with empty string.
  std::string Result;
  // If the vector is not empty
  if (!V.empty()) {
    // Start with the first feature
    Result = V[0];
    // For each successive feature
    for (size_t i = 1; i < V.size(); i++) {
      // Add a comma
      Result += ",";
      // Add the feature
      Result += V[i];
    }
  }
  // Return the features string
  return Result;
}

/// Adding features.
void SubtargetFeatures::AddFeature(const StringRef String,
                                   bool IsEnabled) {
  // Don't add empty features
  if (!String.empty()) {
    // Convert to lowercase, prepend flag and add to vector
    Features.push_back(PrependFlag(String.lower(), IsEnabled));
  }
}

/// Find KV in array using binary search.
static const SubtargetFeatureKV *Find(StringRef S, const SubtargetFeatureKV *A,
                                      size_t L) {
  // Make the lower bound element we're looking for
  SubtargetFeatureKV KV;
  KV.Key = S.data();
  // Determine the end of the array
  const SubtargetFeatureKV *Hi = A + L;
  // Binary search the array
  const SubtargetFeatureKV *F = std::lower_bound(A, Hi, KV);
  // If not found then return NULL
  if (F == Hi || StringRef(F->Key) != S) return NULL;
  // Return the found array item
  return F;
}

/// getLongestEntryLength - Return the length of the longest entry in the table.
///
static size_t getLongestEntryLength(const SubtargetFeatureKV *Table,
                                    size_t Size) {
  size_t MaxLen = 0;
  for (size_t i = 0; i < Size; i++)
    MaxLen = std::max(MaxLen, std::strlen(Table[i].Key));
  return MaxLen;
}

/// Display help for feature choices.
///
static void Help(const SubtargetFeatureKV *CPUTable, size_t CPUTableSize,
                 const SubtargetFeatureKV *FeatTable, size_t FeatTableSize) {
  // Determine the length of the longest CPU and Feature entries.
  unsigned MaxCPULen  = getLongestEntryLength(CPUTable, CPUTableSize);
  unsigned MaxFeatLen = getLongestEntryLength(FeatTable, FeatTableSize);

  // Print the CPU table.
  errs() << "Available CPUs for this target:\n\n";
  for (size_t i = 0; i != CPUTableSize; i++)
    errs() << format("  %-*s - %s.\n",
                     MaxCPULen, CPUTable[i].Key, CPUTable[i].Desc);
  errs() << '\n';

  // Print the Feature table.
  errs() << "Available features for this target:\n\n";
  for (size_t i = 0; i != FeatTableSize; i++)
    errs() << format("  %-*s - %s.\n",
                     MaxFeatLen, FeatTable[i].Key, FeatTable[i].Desc);
  errs() << '\n';

  errs() << "Use +feature to enable a feature, or -feature to disable it.\n"
            "For example, llc -mcpu=mycpu -mattr=+feature1,-feature2\n";
  std::exit(1);
}

//===----------------------------------------------------------------------===//
//                    SubtargetFeatures Implementation
//===----------------------------------------------------------------------===//

SubtargetFeatures::SubtargetFeatures(const StringRef Initial) {
  // Break up string into separate features
  Split(Features, Initial);
}


std::string SubtargetFeatures::getString() const {
  return Join(Features);
}

/// SetImpliedBits - For each feature that is (transitively) implied by this
/// feature, set it.
///
static
void SetImpliedBits(uint64_t &Bits, const SubtargetFeatureKV *FeatureEntry,
                    const SubtargetFeatureKV *FeatureTable,
                    size_t FeatureTableSize) {
  for (size_t i = 0; i < FeatureTableSize; ++i) {
    const SubtargetFeatureKV &FE = FeatureTable[i];

    if (FeatureEntry->Value == FE.Value) continue;

    if (FeatureEntry->Implies & FE.Value) {
      Bits |= FE.Value;
      SetImpliedBits(Bits, &FE, FeatureTable, FeatureTableSize);
    }
  }
}

/// ClearImpliedBits - For each feature that (transitively) implies this
/// feature, clear it.
///
static
void ClearImpliedBits(uint64_t &Bits, const SubtargetFeatureKV *FeatureEntry,
                      const SubtargetFeatureKV *FeatureTable,
                      size_t FeatureTableSize) {
  for (size_t i = 0; i < FeatureTableSize; ++i) {
    const SubtargetFeatureKV &FE = FeatureTable[i];

    if (FeatureEntry->Value == FE.Value) continue;

    if (FE.Implies & FeatureEntry->Value) {
      Bits &= ~FE.Value;
      ClearImpliedBits(Bits, &FE, FeatureTable, FeatureTableSize);
    }
  }
}

/// ToggleFeature - Toggle a feature and returns the newly updated feature
/// bits.
uint64_t
SubtargetFeatures::ToggleFeature(uint64_t Bits, const StringRef Feature,
                                 const SubtargetFeatureKV *FeatureTable,
                                 size_t FeatureTableSize) {
  // Find feature in table.
  const SubtargetFeatureKV *FeatureEntry =
    Find(StripFlag(Feature), FeatureTable, FeatureTableSize);
  // If there is a match
  if (FeatureEntry) {
    if ((Bits & FeatureEntry->Value) == FeatureEntry->Value) {
      Bits &= ~FeatureEntry->Value;

      // For each feature that implies this, clear it.
      ClearImpliedBits(Bits, FeatureEntry, FeatureTable, FeatureTableSize);
    } else {
      Bits |=  FeatureEntry->Value;

      // For each feature that this implies, set it.
      SetImpliedBits(Bits, FeatureEntry, FeatureTable, FeatureTableSize);
    }
  } else {
    errs() << "'" << Feature
           << "' is not a recognized feature for this target"
           << " (ignoring feature)\n";
  }

  return Bits;
}


/// getFeatureBits - Get feature bits a CPU.
///
uint64_t SubtargetFeatures::getFeatureBits(const StringRef CPU,
                                         const SubtargetFeatureKV *CPUTable,
                                         size_t CPUTableSize,
                                         const SubtargetFeatureKV *FeatureTable,
                                         size_t FeatureTableSize) {
  if (!FeatureTableSize || !CPUTableSize)
    return 0;

#ifndef NDEBUG
  for (size_t i = 1; i < CPUTableSize; i++) {
    assert(strcmp(CPUTable[i - 1].Key, CPUTable[i].Key) < 0 &&
           "CPU table is not sorted");
  }
  for (size_t i = 1; i < FeatureTableSize; i++) {
    assert(strcmp(FeatureTable[i - 1].Key, FeatureTable[i].Key) < 0 &&
          "CPU features table is not sorted");
  }
#endif
  uint64_t Bits = 0;                    // Resulting bits

  // Check if help is needed
  if (CPU == "help")
    Help(CPUTable, CPUTableSize, FeatureTable, FeatureTableSize);

  // Find CPU entry if CPU name is specified.
  if (!CPU.empty()) {
    const SubtargetFeatureKV *CPUEntry = Find(CPU, CPUTable, CPUTableSize);
    // If there is a match
    if (CPUEntry) {
      // Set base feature bits
      Bits = CPUEntry->Value;

      // Set the feature implied by this CPU feature, if any.
      for (size_t i = 0; i < FeatureTableSize; ++i) {
        const SubtargetFeatureKV &FE = FeatureTable[i];
        if (CPUEntry->Value & FE.Value)
          SetImpliedBits(Bits, &FE, FeatureTable, FeatureTableSize);
      }
    } else {
      errs() << "'" << CPU
             << "' is not a recognized processor for this target"
             << " (ignoring processor)\n";
    }
  }

  // Iterate through each feature
  for (size_t i = 0, E = Features.size(); i < E; i++) {
    const StringRef Feature = Features[i];

    // Check for help
    if (Feature == "+help")
      Help(CPUTable, CPUTableSize, FeatureTable, FeatureTableSize);

    // Find feature in table.
    const SubtargetFeatureKV *FeatureEntry =
                       Find(StripFlag(Feature), FeatureTable, FeatureTableSize);
    // If there is a match
    if (FeatureEntry) {
      // Enable/disable feature in bits
      if (isEnabled(Feature)) {
        Bits |=  FeatureEntry->Value;

        // For each feature that this implies, set it.
        SetImpliedBits(Bits, FeatureEntry, FeatureTable, FeatureTableSize);
      } else {
        Bits &= ~FeatureEntry->Value;

        // For each feature that implies this, clear it.
        ClearImpliedBits(Bits, FeatureEntry, FeatureTable, FeatureTableSize);
      }
    } else {
      errs() << "'" << Feature
             << "' is not a recognized feature for this target"
             << " (ignoring feature)\n";
    }
  }

  return Bits;
}

/// print - Print feature string.
///
void SubtargetFeatures::print(raw_ostream &OS) const {
  for (size_t i = 0, e = Features.size(); i != e; ++i)
    OS << Features[i] << "  ";
  OS << "\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// dump - Dump feature info.
///
void SubtargetFeatures::dump() const {
  print(dbgs());
}
#endif

/// getDefaultSubtargetFeatures - Return a string listing the features
/// associated with the target triple.
///
/// FIXME: This is an inelegant way of specifying the features of a
/// subtarget. It would be better if we could encode this information
/// into the IR. See <rdar://5972456>.
///
void SubtargetFeatures::getDefaultSubtargetFeatures(const Triple& Triple) {
  if (Triple.getVendor() == Triple::Apple) {
    if (Triple.getArch() == Triple::ppc) {
      // powerpc-apple-*
      AddFeature("altivec");
    } else if (Triple.getArch() == Triple::ppc64) {
      // powerpc64-apple-*
      AddFeature("64bit");
      AddFeature("altivec");
    }
  }
}
