//===- llvm/Testing/Support/Error.h ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TESTING_SUPPORT_ERROR_H
#define LLVM_TESTING_SUPPORT_ERROR_H

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"

#include "gmock/gmock.h"
#include <ostream>

namespace llvm {
namespace detail {
ErrorHolder TakeError(Error Err);

template <typename T> ExpectedHolder<T> TakeExpected(Expected<T> &Exp) {
  return {TakeError(Exp.takeError()), Exp};
}

template <typename T> ExpectedHolder<T> TakeExpected(Expected<T> &&Exp) {
  return TakeExpected(Exp);
}

template <typename T>
class ValueMatchesMono
    : public testing::MatcherInterface<const ExpectedHolder<T> &> {
public:
  explicit ValueMatchesMono(const testing::Matcher<T> &Matcher)
      : Matcher(Matcher) {}

  bool MatchAndExplain(const ExpectedHolder<T> &Holder,
                       testing::MatchResultListener *listener) const override {
    if (!Holder.Success)
      return false;

    bool result = Matcher.MatchAndExplain(*Holder.Exp, listener);

    if (result)
      return result;
    *listener << "(";
    Matcher.DescribeNegationTo(listener->stream());
    *listener << ")";
    return result;
  }

  void DescribeTo(std::ostream *OS) const override {
    *OS << "succeeded with value (";
    Matcher.DescribeTo(OS);
    *OS << ")";
  }

  void DescribeNegationTo(std::ostream *OS) const override {
    *OS << "did not succeed or value (";
    Matcher.DescribeNegationTo(OS);
    *OS << ")";
  }

private:
  testing::Matcher<T> Matcher;
};

template<typename M>
class ValueMatchesPoly {
public:
  explicit ValueMatchesPoly(const M &Matcher) : Matcher(Matcher) {}

  template <typename T>
  operator testing::Matcher<const ExpectedHolder<T> &>() const {
    return MakeMatcher(
        new ValueMatchesMono<T>(testing::SafeMatcherCast<T>(Matcher)));
  }

private:
  M Matcher;
};

} // namespace detail

#define EXPECT_THAT_ERROR(Err, Matcher)                                        \
  EXPECT_THAT(llvm::detail::TakeError(Err), Matcher)
#define ASSERT_THAT_ERROR(Err, Matcher)                                        \
  ASSERT_THAT(llvm::detail::TakeError(Err), Matcher)

#define EXPECT_THAT_EXPECTED(Err, Matcher)                                     \
  EXPECT_THAT(llvm::detail::TakeExpected(Err), Matcher)
#define ASSERT_THAT_EXPECTED(Err, Matcher)                                     \
  ASSERT_THAT(llvm::detail::TakeExpected(Err), Matcher)

MATCHER(Succeeded, "") { return arg.Success; }
MATCHER(Failed, "") { return !arg.Success; }

template <typename M>
detail::ValueMatchesPoly<M> HasValue(M Matcher) {
  return detail::ValueMatchesPoly<M>(Matcher);
}

} // namespace llvm

#endif
