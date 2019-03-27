//===- llvm/MC/SubtargetFeature.h - CPU characteristics ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Defines and manages user or tool specified CPU characteristics.
/// The intent is to be able to package specific features that should or should
/// not be used on a specific target processor.  A tool, such as llc, could, as
/// as example, gather chip info from the command line, a long with features
/// that should be used on that chip.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_SUBTARGETFEATURE_H
#define LLVM_MC_SUBTARGETFEATURE_H

#include "llvm/ADT/StringRef.h"
#include <array>
#include <bitset>
#include <initializer_list>
#include <string>
#include <vector>

namespace llvm {

class raw_ostream;
class Triple;

const unsigned MAX_SUBTARGET_WORDS = 3;
const unsigned MAX_SUBTARGET_FEATURES = MAX_SUBTARGET_WORDS * 64;

/// Container class for subtarget features.
/// This is convenient because std::bitset does not have a constructor
/// with an initializer list of set bits.
class FeatureBitset : public std::bitset<MAX_SUBTARGET_FEATURES> {
public:
  // Cannot inherit constructors because it's not supported by VC++..
  FeatureBitset() = default;

  FeatureBitset(const bitset<MAX_SUBTARGET_FEATURES>& B) : bitset(B) {}

  FeatureBitset(std::initializer_list<unsigned> Init) {
    for (auto I : Init)
      set(I);
  }
};

/// Class used to store the subtarget bits in the tables created by tablegen.
/// The std::initializer_list constructor of FeatureBitset can't be done at
/// compile time and requires a static constructor to run at startup.
class FeatureBitArray {
  std::array<uint64_t, MAX_SUBTARGET_WORDS> Bits;

public:
  constexpr FeatureBitArray(const std::array<uint64_t, MAX_SUBTARGET_WORDS> &B)
      : Bits(B) {}

  FeatureBitset getAsBitset() const {
    FeatureBitset Result;

    for (unsigned i = 0, e = Bits.size(); i != e; ++i)
      Result |= FeatureBitset(Bits[i]) << (64 * i);

    return Result;
  }
};

//===----------------------------------------------------------------------===//

/// Manages the enabling and disabling of subtarget specific features.
///
/// Features are encoded as a string of the form
///   "+attr1,+attr2,-attr3,...,+attrN"
/// A comma separates each feature from the next (all lowercase.)
/// Each of the remaining features is prefixed with + or - indicating whether
/// that feature should be enabled or disabled contrary to the cpu
/// specification.
class SubtargetFeatures {
  std::vector<std::string> Features;    ///< Subtarget features as a vector

public:
  explicit SubtargetFeatures(StringRef Initial = "");

  /// Returns features as a string.
  std::string getString() const;

  /// Adds Features.
  void AddFeature(StringRef String, bool Enable = true);

  /// Returns the vector of individual subtarget features.
  const std::vector<std::string> &getFeatures() const { return Features; }

  /// Prints feature string.
  void print(raw_ostream &OS) const;

  // Dumps feature info.
  void dump() const;

  /// Adds the default features for the specified target triple.
  void getDefaultSubtargetFeatures(const Triple& Triple);

  /// Determine if a feature has a flag; '+' or '-'
  static bool hasFlag(StringRef Feature) {
    assert(!Feature.empty() && "Empty string");
    // Get first character
    char Ch = Feature[0];
    // Check if first character is '+' or '-' flag
    return Ch == '+' || Ch =='-';
  }

  /// Return string stripped of flag.
  static std::string StripFlag(StringRef Feature) {
    return hasFlag(Feature) ? Feature.substr(1) : Feature;
  }

  /// Return true if enable flag; '+'.
  static inline bool isEnabled(StringRef Feature) {
    assert(!Feature.empty() && "Empty string");
    // Get first character
    char Ch = Feature[0];
    // Check if first character is '+' for enabled
    return Ch == '+';
  }

  /// Splits a string of comma separated items in to a vector of strings.
  static void Split(std::vector<std::string> &V, StringRef S);
};

} // end namespace llvm

#endif // LLVM_MC_SUBTARGETFEATURE_H
