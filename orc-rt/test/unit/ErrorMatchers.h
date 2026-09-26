//===- ErrorMatchers.h - gmock matchers for Error/Expected<T> ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// gmock matchers for orc_rt::Error and orc_rt::Expected<T>, after
// llvm/Testing/Support/Error.h.
//
//   EXPECT_THAT_ERROR(mightFail(), Succeeded());
//   EXPECT_THAT_ERROR(mustFail(), Failed<StringError>());
//   EXPECT_THAT_ERROR(mustFail(), FailedWithMessage(HasSubstr("range")));
//   ASSERT_THAT_EXPECTED(compute(), HasValue(42));
//   ASSERT_THAT_EXPECTED(compute(), HasValue(Gt(40)));
//
// An Error argument is consumed, so an lvalue must be moved in:
//
//   ASSERT_THAT_ERROR(std::move(Err), Succeeded());
//
// An Expected<T> argument is not, and keeps its value through a successful
// match:
//
//   ASSERT_THAT_EXPECTED(V, Succeeded());
//   use(*V);
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_UNITTEST_ERRORMATCHERS_H
#define ORC_RT_UNITTEST_ERRORMATCHERS_H

#include "orc-rt/support/Error.h"

#include "gmock/gmock.h"

#include <cassert>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace orc_rt::test {

namespace detail {

/// An Error consumed up front, which matchers then inspect in its place.
///
/// Holds a single ErrorInfoBase: orc_rt has no joinErrors, so an Error carries
/// at most one.
class ErrorHolder {
public:
  ErrorHolder() = default;
  explicit ErrorHolder(std::shared_ptr<ErrorInfoBase> Info)
      : Info(std::move(Info)) {}

  bool success() const noexcept { return Info == nullptr; }

  /// The contained error, which must be present. Non-const through a const
  /// holder, as for the pointer this stands in for.
  ErrorInfoBase &info() const noexcept {
    assert(Info && "No error to inspect");
    return *Info;
  }

private:
  std::shared_ptr<ErrorInfoBase> Info;
};

/// An Expected<T> consumed the same way, keeping a reference to the original so
/// a matcher can reach the value.
template <typename T> class ExpectedHolder : public ErrorHolder {
public:
  ExpectedHolder(ErrorHolder Err, Expected<T> &Exp)
      : ErrorHolder(std::move(Err)), Exp(Exp) {}

  Expected<T> &Exp;
};

/// Renders a holder for the "Actual:" line of a failure. Found by gmock through
/// ADL.
inline void PrintTo(const ErrorHolder &Err, std::ostream *Out) {
  if (Err.success()) {
    *Out << "succeeded";
    return;
  }
  *Out << "failed with " << Err.info().dynamicRTTIName() << ": "
       << Err.info().toString();
}

template <typename T>
void PrintTo(const ExpectedHolder<T> &Item, std::ostream *Out) {
  if (!Item.success()) {
    PrintTo(static_cast<const ErrorHolder &>(Item), Out);
    return;
  }
  *Out << "succeeded with value " << ::testing::PrintToString(*Item.Exp);
}

inline ErrorHolder takeError(Error Err) {
  std::shared_ptr<ErrorInfoBase> Info;
  handleAllErrors(std::move(Err), [&Info](std::unique_ptr<ErrorInfoBase> I) {
    Info = std::move(I);
  });
  return ErrorHolder(std::move(Info));
}

template <typename T> ExpectedHolder<T> takeExpected(Expected<T> &Exp) {
  return ExpectedHolder<T>(takeError(Exp.takeError()), Exp);
}

template <typename T> ExpectedHolder<T> takeExpected(Expected<T> &&Exp) {
  return takeExpected(Exp);
}

/// Matches a failure Error whose contained error is of type InfoT, and which
/// additionally satisfies Matcher if one is given.
template <typename InfoT>
class ErrorMatchesMono
    : public ::testing::MatcherInterface<const ErrorHolder &> {
public:
  explicit ErrorMatchesMono(std::optional<::testing::Matcher<InfoT &>> Matcher)
      : Matcher(std::move(Matcher)) {}

  bool
  MatchAndExplain(const ErrorHolder &Holder,
                  ::testing::MatchResultListener *Listener) const override {
    if (Holder.success()) {
      *Listener << "succeeded";
      return false;
    }

    ErrorInfoBase &Info = Holder.info();
    if (!Info.isA<InfoT>()) {
      *Listener << "failed with " << Info.dynamicRTTIName() << ": "
                << Info.toString();
      return false;
    }

    if (!Matcher)
      return true;

    return Matcher->MatchAndExplain(static_cast<InfoT &>(Info), Listener);
  }

  void DescribeTo(std::ostream *OS) const override {
    *OS << "failed with " << InfoT::RTTIName;
    if (Matcher) {
      *OS << " and the error ";
      Matcher->DescribeTo(OS);
    }
  }

  void DescribeNegationTo(std::ostream *OS) const override {
    *OS << "succeeded, or failed with something other than " << InfoT::RTTIName;
    if (Matcher) {
      *OS << ", or the error ";
      Matcher->DescribeNegationTo(OS);
    }
  }

private:
  std::optional<::testing::Matcher<InfoT &>> Matcher;
};

/// Matches a failure Error whose message satisfies Matcher.
class ErrorMessageMatches
    : public ::testing::MatcherInterface<const ErrorHolder &> {
public:
  explicit ErrorMessageMatches(::testing::Matcher<std::string> Matcher)
      : Matcher(std::move(Matcher)) {}

  bool
  MatchAndExplain(const ErrorHolder &Holder,
                  ::testing::MatchResultListener *Listener) const override {
    if (Holder.success()) {
      *Listener << "succeeded";
      return false;
    }
    return Matcher.MatchAndExplain(Holder.info().toString(), Listener);
  }

  void DescribeTo(std::ostream *OS) const override {
    *OS << "failed with an error whose message ";
    Matcher.DescribeTo(OS);
  }

  void DescribeNegationTo(std::ostream *OS) const override {
    *OS << "succeeded, or failed with an error whose message ";
    Matcher.DescribeNegationTo(OS);
  }

private:
  ::testing::Matcher<std::string> Matcher;
};

/// Matches a success Expected<T> whose value satisfies Matcher.
template <typename T>
class ValueMatchesMono
    : public ::testing::MatcherInterface<const ExpectedHolder<T> &> {
public:
  explicit ValueMatchesMono(const ::testing::Matcher<T> &Matcher)
      : Matcher(Matcher) {}

  bool
  MatchAndExplain(const ExpectedHolder<T> &Holder,
                  ::testing::MatchResultListener *Listener) const override {
    if (!Holder.success()) {
      *Listener << "failed with " << Holder.info().dynamicRTTIName() << ": "
                << Holder.info().toString();
      return false;
    }

    if (Matcher.MatchAndExplain(*Holder.Exp, Listener))
      return true;

    if (Listener->IsInterested()) {
      *Listener << "(";
      Matcher.DescribeNegationTo(Listener->stream());
      *Listener << ")";
    }
    return false;
  }

  void DescribeTo(std::ostream *OS) const override {
    *OS << "succeeded with value (";
    Matcher.DescribeTo(OS);
    *OS << ")";
  }

  void DescribeNegationTo(std::ostream *OS) const override {
    *OS << "did not succeed, or value (";
    Matcher.DescribeNegationTo(OS);
    *OS << ")";
  }

private:
  ::testing::Matcher<T> Matcher;
};

/// Defers naming T until the ExpectedHolder<T> being matched is known, so
/// HasValue(42) needs no explicit value type.
template <typename M> class ValueMatchesPoly {
public:
  explicit ValueMatchesPoly(const M &Matcher) : Matcher(Matcher) {}

  template <typename T>
  operator ::testing::Matcher<const ExpectedHolder<T> &>() const {
    return ::testing::MakeMatcher(
        new ValueMatchesMono<T>(::testing::SafeMatcherCast<T>(Matcher)));
  }

private:
  M Matcher;
};

} // namespace detail

#define EXPECT_THAT_ERROR(Err, Matcher)                                        \
  EXPECT_THAT(::orc_rt::test::detail::takeError(Err), Matcher)
#define ASSERT_THAT_ERROR(Err, Matcher)                                        \
  ASSERT_THAT(::orc_rt::test::detail::takeError(Err), Matcher)

#define EXPECT_THAT_EXPECTED(Val, Matcher)                                     \
  EXPECT_THAT(::orc_rt::test::detail::takeExpected(Val), Matcher)
#define ASSERT_THAT_EXPECTED(Val, Matcher)                                     \
  ASSERT_THAT(::orc_rt::test::detail::takeExpected(Val), Matcher)

/// Matches an Error or Expected<T> holding no error.
MATCHER(Succeeded, "") { return arg.success(); }

/// Matches an Error or Expected<T> holding an error, of any type.
MATCHER(Failed, "") { return !arg.success(); }

/// Matches an Error holding an error of type InfoT.
template <typename InfoT>
::testing::Matcher<const detail::ErrorHolder &> Failed() {
  return ::testing::MakeMatcher(
      new detail::ErrorMatchesMono<InfoT>(std::nullopt));
}

/// Matches an Error holding an error of type InfoT that satisfies Matcher.
template <typename InfoT, typename M>
::testing::Matcher<const detail::ErrorHolder &> Failed(M Matcher) {
  return ::testing::MakeMatcher(new detail::ErrorMatchesMono<InfoT>(
      ::testing::SafeMatcherCast<InfoT &>(Matcher)));
}

/// Matches an Error holding an error whose message satisfies Matcher. A bare
/// string means exact equality.
template <typename M>
::testing::Matcher<const detail::ErrorHolder &> FailedWithMessage(M Matcher) {
  return ::testing::MakeMatcher(new detail::ErrorMessageMatches(Matcher));
}

/// Matches an Expected<T> holding a value that satisfies Matcher.
template <typename M> detail::ValueMatchesPoly<M> HasValue(M Matcher) {
  return detail::ValueMatchesPoly<M>(Matcher);
}

} // namespace orc_rt::test

#endif // ORC_RT_UNITTEST_ERRORMATCHERS_H
