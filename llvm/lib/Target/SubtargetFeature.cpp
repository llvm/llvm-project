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

#include "llvm/Target/SubtargetFeature.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Streams.h"
#include <algorithm>
#include <ostream>
#include <cassert>
#include <cctype>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                          Static Helper Functions
//===----------------------------------------------------------------------===//

/// hasFlag - Determine if a feature has a flag; '+' or '-'
///
static inline bool hasFlag(const std::string &Feature) {
  assert(!Feature.empty() && "Empty string");
  // Get first character
  char Ch = Feature[0];
  // Check if first character is '+' or '-' flag
  return Ch == '+' || Ch =='-';
}

/// StripFlag - Return string stripped of flag.
///
static inline std::string StripFlag(const std::string &Feature) {
  return hasFlag(Feature) ? Feature.substr(1) : Feature;
}

/// isEnabled - Return true if enable flag; '+'.
///
static inline bool isEnabled(const std::string &Feature) {
  assert(!Feature.empty() && "Empty string");
  // Get first character
  char Ch = Feature[0];
  // Check if first character is '+' for enabled
  return Ch == '+';
}

/// PrependFlag - Return a string with a prepended flag; '+' or '-'.
///
static inline std::string PrependFlag(const std::string &Feature,
                                      bool IsEnabled) {
  assert(!Feature.empty() && "Empty string");
  if (hasFlag(Feature)) return Feature;
  return std::string(IsEnabled ? "+" : "-") + Feature;
}

/// Split - Splits a string of comma separated items in to a vector of strings.
///
static void Split(std::vector<std::string> &V, const std::string &S) {
  // Start at beginning of string.
  size_t Pos = 0;
  while (true) {
    // Find the next comma
    size_t Comma = S.find(',', Pos);
    // If no comma found then the the rest of the string is used
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
    // Start with the CPU feature
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
void SubtargetFeatures::AddFeature(const std::string &String,
                                   bool IsEnabled) {
  // Don't add empty features
  if (!String.empty()) {
    // Convert to lowercase, prepend flag and add to vector
    Features.push_back(PrependFlag(LowercaseString(String), IsEnabled));
  }
}

/// Find KV in array using binary search.
template<typename T> const T *Find(const std::string &S, const T *A, size_t L) {
  // Make the lower bound element we're looking for
  T KV;
  KV.Key = S.c_str();
  // Determine the end of the array
  const T *Hi = A + L;
  // Binary search the array
  const T *F = std::lower_bound(A, Hi, KV);
  // If not found then return NULL
  if (F == Hi || std::string(F->Key) != S) return NULL;
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
  cerr << "Available CPUs for this target:\n\n";
  for (size_t i = 0; i != CPUTableSize; i++)
    cerr << "  " << CPUTable[i].Key
         << std::string(MaxCPULen - std::strlen(CPUTable[i].Key), ' ')
         << " - " << CPUTable[i].Desc << ".\n";
  cerr << "\n";
  
  // Print the Feature table.
  cerr << "Available features for this target:\n\n";
  for (size_t i = 0; i != FeatTableSize; i++)
    cerr << "  " << FeatTable[i].Key
         << std::string(MaxFeatLen - std::strlen(FeatTable[i].Key), ' ')
         << " - " << FeatTable[i].Desc << ".\n";
  cerr << "\n";
  
  cerr << "Use +feature to enable a feature, or -feature to disable it.\n"
       << "For example, llc -mcpu=mycpu -mattr=+feature1,-feature2\n";
  exit(1);
}

//===----------------------------------------------------------------------===//
//                    SubtargetFeatures Implementation
//===----------------------------------------------------------------------===//

SubtargetFeatures::SubtargetFeatures(const std::string &Initial) {
  // Break up string into separate features
  Split(Features, Initial);
}


std::string SubtargetFeatures::getString() const {
  return Join(Features);
}
void SubtargetFeatures::setString(const std::string &Initial) {
  // Throw out old features
  Features.clear();
  // Break up string into separate features
  Split(Features, LowercaseString(Initial));
}


/// setCPU - Set the CPU string.  Replaces previous setting.  Setting to "" 
/// clears CPU.
void SubtargetFeatures::setCPU(const std::string &String) {
  Features[0] = LowercaseString(String);
}


/// setCPUIfNone - Setting CPU string only if no string is set.
///
void SubtargetFeatures::setCPUIfNone(const std::string &String) {
  if (Features[0].empty()) setCPU(String);
}

/// SetImpliedBits - For each feature that is (transitively) implied by this
/// feature, set it.
/// 
static
void SetImpliedBits(uint32_t &Bits, const SubtargetFeatureKV *FeatureEntry,
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
void ClearImpliedBits(uint32_t &Bits, const SubtargetFeatureKV *FeatureEntry,
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

/// getBits - Get feature bits.
///
uint32_t SubtargetFeatures::getBits(const SubtargetFeatureKV *CPUTable,
                                          size_t CPUTableSize,
                                    const SubtargetFeatureKV *FeatureTable,
                                          size_t FeatureTableSize) {
  assert(CPUTable && "missing CPU table");
  assert(FeatureTable && "missing features table");
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
  uint32_t Bits = 0;                    // Resulting bits

  // Check if help is needed
  if (Features[0] == "help")
    Help(CPUTable, CPUTableSize, FeatureTable, FeatureTableSize);
  
  // Find CPU entry
  const SubtargetFeatureKV *CPUEntry =
                            Find(Features[0], CPUTable, CPUTableSize);
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
    cerr << "'" << Features[0]
         << "' is not a recognized processor for this target"
         << " (ignoring processor)"
         << "\n";
  }
  // Iterate through each feature
  for (size_t i = 1; i < Features.size(); i++) {
    const std::string &Feature = Features[i];
    
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
      cerr << "'" << Feature
           << "' is not a recognized feature for this target"
           << " (ignoring feature)"
           << "\n";
    }
  }

  return Bits;
}

/// Get info pointer
void *SubtargetFeatures::getInfo(const SubtargetInfoKV *Table,
                                       size_t TableSize) {
  assert(Table && "missing table");
#ifndef NDEBUG
  for (size_t i = 1; i < TableSize; i++) {
    assert(strcmp(Table[i - 1].Key, Table[i].Key) < 0 && "Table is not sorted");
  }
#endif

  // Find entry
  const SubtargetInfoKV *Entry = Find(Features[0], Table, TableSize);
  
  if (Entry) {
    return Entry->Value;
  } else {
    cerr << "'" << Features[0]
         << "' is not a recognized processor for this target"
         << " (ignoring processor)"
         << "\n";
    return NULL;
  }
}

/// print - Print feature string.
///
void SubtargetFeatures::print(std::ostream &OS) const {
  for (size_t i = 0; i < Features.size(); i++) {
    OS << Features[i] << "  ";
  }
  OS << "\n";
}

/// dump - Dump feature info.
///
void SubtargetFeatures::dump() const {
  print(*cerr.stream());
}
