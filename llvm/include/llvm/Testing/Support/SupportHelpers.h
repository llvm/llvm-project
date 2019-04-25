//===- Testing/Support/SupportHelpers.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TESTING_SUPPORT_SUPPORTHELPERS_H
#define LLVM_TESTING_SUPPORT_SUPPORTHELPERS_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest-printers.h"

#include <string>

namespace llvm {
namespace detail {
struct ErrorHolder {
  std::vector<std::shared_ptr<ErrorInfoBase>> Infos;

  bool Success() const { return Infos.empty(); }
};

template <typename T> struct ExpectedHolder : public ErrorHolder {
  ExpectedHolder(ErrorHolder Err, Expected<T> &Exp)
      : ErrorHolder(std::move(Err)), Exp(Exp) {}

  Expected<T> &Exp;
};

inline void PrintTo(const ErrorHolder &Err, std::ostream *Out) {
  raw_os_ostream OS(*Out);
  OS << (Err.Success() ? "succeeded" : "failed");
  if (!Err.Success()) {
    const char *Delim = "  (";
    for (const auto &Info : Err.Infos) {
      OS << Delim;
      Delim = "; ";
      Info->log(OS);
    }
    OS << ")";
  }
}

template <typename T>
void PrintTo(const ExpectedHolder<T> &Item, std::ostream *Out) {
  if (Item.Success()) {
    *Out << "succeeded with value " << ::testing::PrintToString(*Item.Exp);
  } else {
    PrintTo(static_cast<const ErrorHolder &>(Item), Out);
  }
}

template <class InnerMatcher> class ValueIsMatcher {
public:
  explicit ValueIsMatcher(InnerMatcher ValueMatcher)
      : ValueMatcher(ValueMatcher) {}

  template <class T>
  operator ::testing::Matcher<const llvm::Optional<T> &>() const {
    return ::testing::MakeMatcher(
        new Impl<T>(::testing::SafeMatcherCast<T>(ValueMatcher)));
  }

  template <class T>
  class Impl : public ::testing::MatcherInterface<const llvm::Optional<T> &> {
  public:
    explicit Impl(const ::testing::Matcher<T> &ValueMatcher)
        : ValueMatcher(ValueMatcher) {}

    bool MatchAndExplain(const llvm::Optional<T> &Input,
                         testing::MatchResultListener *L) const override {
      return Input && ValueMatcher.MatchAndExplain(Input.getValue(), L);
    }

    void DescribeTo(std::ostream *OS) const override {
      *OS << "has a value that ";
      ValueMatcher.DescribeTo(OS);
    }
    void DescribeNegationTo(std::ostream *OS) const override {
      *OS << "does not have a value that ";
      ValueMatcher.DescribeTo(OS);
    }

  private:
    testing::Matcher<T> ValueMatcher;
  };

private:
  InnerMatcher ValueMatcher;
};
} // namespace detail

/// Matches an llvm::Optional<T> with a value that conforms to an inner matcher.
/// To match llvm::None you could use Eq(llvm::None).
template <class InnerMatcher>
detail::ValueIsMatcher<InnerMatcher> ValueIs(const InnerMatcher &ValueMatcher) {
  return detail::ValueIsMatcher<InnerMatcher>(ValueMatcher);
}
namespace unittest {
SmallString<128> getInputFileDirectory(const char *Argv0);
} // namespace unittest
} // namespace llvm

#endif
