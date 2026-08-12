//===- DwarfTransformer.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_GSYM_OUTPUTAGGREGATOR_H
#define LLVM_DEBUGINFO_GSYM_OUTPUTAGGREGATOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/GSYM/ExtractRanges.h"

#include <map>
#include <string>

namespace llvm {

class raw_ostream;

namespace gsym {

// How much of the debug information diagnostic output to suppress. Each level
// suppresses everything the previous one does, so these are ordered and are
// meant to be compared with >=.
enum class QuietLevel : uint8_t {
  None = 0, // Emit warnings and errors.
  Quiet,    // Emit errors only.
  Quieter,  // Emit neither warnings nor errors.
};

// The severity of a diagnostic handed to OutputAggregator::Report().
enum class Severity : uint8_t { Warning, Error };

class OutputAggregator {
protected:
  // A std::map is preferable over an llvm::StringMap for presenting results
  // in a predictable order.
  std::map<std::string, unsigned> Aggregation;
  raw_ostream *Out;
  // Diagnostics silenced by this level are still counted, so the aggregated
  // totals are the same no matter how quiet we are; only the detail messages
  // and anything written through operator<< are affected.
  QuietLevel Quiet;

public:
  OutputAggregator(raw_ostream *out, QuietLevel Quiet = QuietLevel::None)
      : Out(out), Quiet(Quiet) {}

  size_t GetNumCategories() const { return Aggregation.size(); }

  QuietLevel GetQuietLevel() const { return Quiet; }

  // Returns true if the detail message for a diagnostic of this severity
  // should be silenced.
  bool IsSuppressed(Severity Sev) const {
    return Sev == Severity::Error ? Quiet >= QuietLevel::Quieter
                                  : Quiet >= QuietLevel::Quiet;
  }

  void Report(StringRef s, Severity Sev,
              std::function<void(raw_ostream &o)> detailCallback) {
    Aggregation[std::string(s)]++;
    if (GetOS() && !IsSuppressed(Sev))
      detailCallback(*Out);
  }

  void EnumerateResults(
      std::function<void(StringRef, unsigned)> handleCounts) const {
    for (auto &&[name, count] : Aggregation)
      handleCounts(name, count);
  }

  raw_ostream *GetOS() const { return Out; }

  // You can just use the stream, and if it's null, nothing happens.
  // Don't do a lot of stuff like this, but it's convenient for silly stuff.
  // It doesn't work with things that have custom insertion operators, though.
  template <typename T> OutputAggregator &operator<<(T &&value) {
    if (Out != nullptr)
      *Out << value;
    return *this;
  }

  // For multi-threaded usage, we can collect stuff in another aggregator,
  // then merge it in here. Note that this is *not* thread safe. It is up to
  // the caller to ensure that this is only called from one thread at a time.
  void Merge(const OutputAggregator &other) {
    for (auto &&[name, count] : other.Aggregation)
      Aggregation[name] += count;
  }
};

} // namespace gsym
} // namespace llvm

#endif // LLVM_DEBUGINFO_GSYM_OUTPUTAGGREGATOR_H
