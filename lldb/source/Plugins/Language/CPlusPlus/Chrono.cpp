//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Chrono.h"

#include "lldb/ValueObject/ValueObject.h"

#include <ctime>

using namespace lldb;
using namespace lldb_private;

namespace {

/// Helper to get members of a type from different STLs.
///
/// The member names of the different STL types are expected to be unique.
///
/// \tparam N Number of members to get.
template <size_t N = 1> struct TryStl {
  static_assert(N > 0);

  using Array = ValueObjectSP[N];

  explicit TryStl(ValueObject &source) : source(source) {}
  TryStl(const TryStl &) = delete;
  TryStl(TryStl &&) = delete;
  TryStl &operator=(const TryStl &) = delete;
  TryStl &operator=(TryStl &&) = delete;

  template <typename... Str> TryStl &Libcxx(Str... strs) {
    return Try(std::forward<Str>(strs)...);
  }
  template <typename... Str> TryStl &Libstdcxx(Str... strs) {
    return Try(std::forward<Str>(strs)...);
  }
  template <typename... Str> TryStl &Msvc(Str... strs) {
    return Try(std::forward<Str>(strs)...);
  }

  ValueObjectSP TakeFirst() { return std::move(valobjs[0]); }

  Array &&All() { return std::move(valobjs); }

private:
  template <typename... Str> TryStl &Try(llvm::StringRef first, Str... rest) {
    if (done)
      return *this;
    static_assert(sizeof...(rest) + 1 == N, "Must provide exactly N strings");

    // Only check the first member - if this matches, we can get the rest.
    valobjs[0] = source.GetChildMemberWithName(first);
    if (!valobjs[0])
      return *this;

    done = true;
    size_t i = 1;
    ((valobjs[i++] = source.GetChildMemberWithName(rest)), ...);
    return *this;
  }

  bool done = false;
  ValueObject &source;
  ValueObjectSP valobjs[N]{};
};

TryStl(ValueObject &) -> TryStl<1>;

ValueObjectSP TryOne(ValueObject &source, llvm::StringRef libcxx,
                     llvm::StringRef libstdcxx, llvm::StringRef msvc) {
  return TryStl(source)
      .Libcxx(libcxx)
      .Libstdcxx(libstdcxx)
      .Msvc(msvc)
      .TakeFirst();
}

bool ChronoDurationFormat(ValueObject &valobj, Stream &stream,
                          llvm::StringRef unit) {
  ValueObjectSP rep_sp = TryOne(valobj, "__rep_", "__r", "_MyRep");
  if (!rep_sp)
    return false;
  int64_t rep = rep_sp->GetValueAsSigned(0);
  stream << llvm::formatv("{0} {1}", rep, unit);
  return true;
}

std::optional<int64_t> TimePointDuration(ValueObject &valobj) {
  ValueObjectSP dur_sp = TryOne(valobj, "__d_", "__d", "_MyDur");
  if (!dur_sp)
    return std::nullopt;
  dur_sp = dur_sp->GetChildAtIndex(0); // Rep
  if (!dur_sp)
    return std::nullopt;
  return dur_sp->GetValueAsSigned(0);
}

bool TimePointSummary(ValueObject &valobj, Stream &stream, std::time_t to_sec,
                      const char *fmt, const char *suffix) {
  std::optional<int64_t> tp_duration = TimePointDuration(valobj);
  if (!tp_duration)
    return false;

#ifndef _WIN32
  // The date time in the chrono library is valid in the range
  // [-32767-01-01T00:00:00Z, 32767-12-31T23:59:59Z]. A 64-bit time_t has a
  // larger range, the function strftime is not able to format the entire range
  // of time_t. The exact point has not been investigated; it's limited to
  // chrono's range.
  const std::time_t chrono_timestamp_min =
      -1'096'193'779'200; // -32767-01-01T00:00:00Z
  const std::time_t chrono_timestamp_max =
      971'890'963'199; // 32767-12-31T23:59:59Z
#else
  const std::time_t chrono_timestamp_min = -43'200; // 1969-12-31T12:00:00Z
  const std::time_t chrono_timestamp_max =
      32'536'850'399; // 3001-01-19T21:59:59
#endif

  const std::time_t seconds = *tp_duration * to_sec;
  if (seconds < chrono_timestamp_min || seconds > chrono_timestamp_max)
    stream.Printf("timestamp=%" PRId64 " %s", *tp_duration, suffix);
  else {
    std::array<char, 128> str;
    std::size_t size =
        std::strftime(str.data(), str.size(), fmt, gmtime(&seconds));
    if (size == 0)
      return false;

    const char *label = to_sec < (60 * 60 * 24) ? "date/time" : "date";

    stream.Printf("%s=%s timestamp=%" PRId64 " %s", label, str.data(),
                  *tp_duration, suffix);
  }

  return true;
}

} // namespace

namespace lldb_private::formatters::chrono {

#pragma region Calendar

/// std::chrono::day
bool DaySummaryProvider(ValueObject &valobj, Stream &stream,
                        const TypeSummaryOptions &options) {
  ValueObjectSP day_sp = TryOne(valobj, "__d_", "_M_d", "_Day");
  if (!day_sp)
    return false;

  stream << llvm::formatv("day={0}", day_sp->GetValueAsUnsigned(0));
  return true;
}

/// std::chrono::month
bool MonthSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options) {
  // FIXME: These are the names used in the C++20 ostream operator. Since LLVM
  // uses C++17 it's not possible to use the ostream operator directly.
  static const std::array<std::string_view, 12> months = {
      "January", "February", "March",     "April",   "May",      "June",
      "July",    "August",   "September", "October", "November", "December"};

  ValueObjectSP ptr_sp = TryOne(valobj, "__m_", "_M_m", "_Month");
  if (!ptr_sp)
    return false;

  const unsigned month = ptr_sp->GetValueAsUnsigned(0);
  if (month >= 1 && month <= 12)
    stream << "month=" << months[month - 1];
  else
    stream << llvm::formatv("month={0}", month);

  return true;
}

/// std::chrono::year
bool YearSummaryProvider(ValueObject &valobj, Stream &stream,
                         const TypeSummaryOptions &options) {
  ValueObjectSP year_sp = TryOne(valobj, "__y_", "_M_y", "_Year");
  if (!year_sp)
    return false;

  stream << llvm::formatv("year={0}", year_sp->GetValueAsSigned(0));
  return true;
}

/// std::chrono::weekday
bool WeekdaySummaryProvider(ValueObject &valobj, Stream &stream,
                            const TypeSummaryOptions &options) {
  // FIXME: These are the names used in the C++20 ostream operator. Since LLVM
  // uses C++17 it's not possible to use the ostream operator directly.
  static const std::array<std::string_view, 7> weekdays = {
      "Sunday",   "Monday", "Tuesday", "Wednesday",
      "Thursday", "Friday", "Saturday"};

  ValueObjectSP ptr_sp = TryOne(valobj, "__wd_", "_M_wd", "_Weekday");
  if (!ptr_sp)
    return false;

  const unsigned weekday = ptr_sp->GetValueAsUnsigned(0);
  if (weekday < 7)
    stream << "weekday=" << weekdays[weekday];
  else
    stream.Printf("weekday=%u", weekday);

  return true;
}

/// std::chrono::weekday_indexed
bool WeekdayIndexedSummaryProvider(ValueObject &valobj, Stream &stream,
                                   const TypeSummaryOptions &options) {
  auto [weekday_sp, idx_sp] = TryStl<2>(valobj)
                                  .Libcxx("__wd_", "__idx_")
                                  .Libstdcxx("_M_wd", "_M_index")
                                  .Msvc("_Weekday", "_Index")
                                  .All();
  if (!weekday_sp || !idx_sp)
    return false;

  stream << llvm::formatv("{0} index={1}", weekday_sp->GetSummaryAsCString(),
                          idx_sp->GetValueAsUnsigned(0));
  return true;
}

/// std::chrono::weekday_last
bool WeekdayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &options) {
  ValueObjectSP weekday_sp = TryOne(valobj, "__wd_", "_M_wd", "_Weekday");
  if (!weekday_sp)
    return false;

  stream << llvm::formatv("{0} index=last", weekday_sp->GetSummaryAsCString());
  return true;
}

/// std::chrono::month_day
bool MonthDaySummaryProvider(ValueObject &valobj, Stream &stream,
                             const TypeSummaryOptions &options) {
  auto [month_sp, day_sp] = TryStl<2>(valobj)
                                .Libcxx("__m_", "__d_")
                                .Libstdcxx("_M_m", "_M_d")
                                .Msvc("_Month", "_Day")
                                .All();
  if (!month_sp || !day_sp)
    return false;

  stream << llvm::formatv("{0} {1}", month_sp->GetSummaryAsCString(),
                          day_sp->GetSummaryAsCString());
  return true;
}

/// std::chrono::month_day_last
bool MonthDayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options) {
  ValueObjectSP month_sp = TryOne(valobj, "__m_", "_M_m", "_Month");
  if (!month_sp)
    return false;

  stream << llvm::formatv("{0} day=last", month_sp->GetSummaryAsCString());
  return true;
}

/// std::chrono::month_weekday
bool MonthWeekdaySummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options) {
  auto [month_sp, weekday_sp] = TryStl<2>(valobj)
                                    .Libcxx("__m_", "__wdi_")
                                    .Libstdcxx("_M_m", "_M_wdi")
                                    .Msvc("_Month", "_Weekday_index")
                                    .All();
  if (!month_sp || !weekday_sp)
    return false;

  stream << llvm::formatv("{0} {1}", month_sp->GetSummaryAsCString(),
                          weekday_sp->GetSummaryAsCString());
  return true;
}

/// std::chrono::month_weekday_last
bool MonthWeekdayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                     const TypeSummaryOptions &options) {
  auto [month_sp, weekday_sp] = TryStl<2>(valobj)
                                    .Libcxx("__m_", "__wdl_")
                                    .Libstdcxx("_M_m", "_M_wdl")
                                    .Msvc("_Month", "_Weekday_last")
                                    .All();
  if (!month_sp || !weekday_sp)
    return false;

  stream << llvm::formatv("{0} {1}", month_sp->GetSummaryAsCString(),
                          weekday_sp->GetSummaryAsCString());
  return true;
}

/// std::chrono::year_month
bool YearMonthSummaryProvider(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &options) {
  auto [year_sp, month_sp] = TryStl<2>(valobj)
                                 .Libcxx("__y_", "__m_")
                                 .Libstdcxx("_M_y", "_M_m")
                                 .Msvc("_Year", "_Month")
                                 .All();
  if (!year_sp || !month_sp)
    return false;

  stream << llvm::formatv("{0} {1}", year_sp->GetSummaryAsCString(),
                          month_sp->GetSummaryAsCString());
  return true;
}

/// std::chrono::year_month_day
bool YearMonthDaySummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options) {
  auto [year_sp, month_sp, day_sp] = TryStl<3>(valobj)
                                         .Libcxx("__y_", "__m_", "__d_")
                                         .Libstdcxx("_M_y", "_M_m", "_M_d")
                                         .Msvc("_Year", "_Month", "_Day")
                                         .All();
  if (!year_sp || !month_sp || !day_sp)
    return false;
  year_sp = year_sp->GetChildAtIndex(0);
  month_sp = month_sp->GetChildAtIndex(0);
  day_sp = day_sp->GetChildAtIndex(0);
  if (!year_sp || !month_sp || !day_sp)
    return false;

  int64_t year = year_sp->GetValueAsSigned(0);
  uint64_t month = month_sp->GetValueAsUnsigned(0);
  uint64_t day = day_sp->GetValueAsUnsigned(0);

  stream << "date=";
  if (year < 0) {
    stream << '-';
    year = -year;
  }
  stream << llvm::formatv("{0,0+4}-{1,0+2}-{2,0+2}", year, month, day);
  return true;
}

/// std::chrono::year_month_day_last
bool YearMonthDayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                     const TypeSummaryOptions &options) {
  auto [year_sp, month_sp] = TryStl<2>(valobj)
                                 .Libcxx("__y_", "__mdl_")
                                 .Libstdcxx("_M_y", "_M_mdl")
                                 .Msvc("_Year", "_Month_day_last")
                                 .All();
  if (!year_sp || !month_sp)
    return false;

  stream << llvm::formatv("{0} {1}", year_sp->GetSummaryAsCString(),
                          month_sp->GetSummaryAsCString());
  return true;
}

/// std::chrono::year_month_weekday
bool YearMonthWeekdaySummaryProvider(ValueObject &valobj, Stream &stream,
                                     const TypeSummaryOptions &options) {
  auto [year_sp, month_sp, wdi_sp] =
      TryStl<3>(valobj)
          .Libcxx("__y_", "__m_", "__wdi_")
          .Libstdcxx("_M_y", "_M_m", "_M_wdi")
          .Msvc("_Year", "_Month", "_Weekday_index")
          .All();
  if (!year_sp || !month_sp || !wdi_sp)
    return false;

  stream << llvm::formatv("{0} {1} {2}", year_sp->GetSummaryAsCString(),
                          month_sp->GetSummaryAsCString(),
                          wdi_sp->GetSummaryAsCString());
  return true;
}

/// std::chrono::year_month_weekday_last
bool YearMonthWeekdayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                         const TypeSummaryOptions &options) {
  auto [year_sp, month_sp, wdl_sp] =
      TryStl<3>(valobj)
          .Libcxx("__y_", "__m_", "__wdl_")
          .Libstdcxx("_M_y", "_M_m", "_M_wdl")
          .Msvc("_Year", "_Month", "_Weekday_last")
          .All();
  if (!year_sp || !month_sp || !wdl_sp)
    return false;

  stream << llvm::formatv("{0} {1} {2}", year_sp->GetSummaryAsCString(),
                          month_sp->GetSummaryAsCString(),
                          wdl_sp->GetSummaryAsCString());
  return true;
}

#pragma region Durations

/// std::chrono::nanoseconds
bool NanosecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "ns");
}

/// std::chrono::microseconds
bool MicrosecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "µs");
}

/// std::chrono::milliseconds
bool MillisecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "ms");
}

/// std::chrono::seconds
bool SecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                            const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "s");
}

/// std::chrono::minutes
bool MinutesSummaryProvider(ValueObject &valobj, Stream &stream,
                            const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "min");
}

/// std::chrono::hours
bool HoursSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "h");
}

/// std::chrono::days
bool DaysSummaryProvider(ValueObject &valobj, Stream &stream,
                         const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "days");
}

/// std::chrono::weeks
bool WeeksSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "weeks");
}

/// std::chrono::years
bool YearsSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "years");
}

/// std::chrono::months
bool MonthsSummaryProvider(ValueObject &valobj, Stream &stream,
                           const TypeSummaryOptions &options) {
  return ChronoDurationFormat(valobj, stream, "months");
}

#pragma region Timepoints

/// std::chrono::time_point<system_clock, seconds>
bool SysSecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                               const TypeSummaryOptions &options) {
  return TimePointSummary(valobj, stream, /*to_sec=*/1, "%FT%H:%M:%SZ",
                          /*suffix=*/"s");
}

/// std::chrono::time_point<system_clock, days>
bool SysDaysSummaryProvider(ValueObject &valobj, Stream &stream,
                            const TypeSummaryOptions &options) {
  return TimePointSummary(valobj, stream, /*to_sec=*/60 * 60 * 24, "%FZ",
                          /*suffix=*/"days");
}

/// std::chrono::time_point<local_t, seconds>
bool LocalSecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options) {
  return TimePointSummary(valobj, stream, /*to_sec=*/1, "%FT%H:%M:%S",
                          /*suffix=*/"s");
}

/// std::chrono::time_point<local_t, days>
bool LocalDaysSummaryProvider(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &options) {
  return TimePointSummary(valobj, stream, /*to_sec=*/60 * 60 * 24, "%F",
                          /*suffix=*/"days");
}

} // namespace lldb_private::formatters::chrono
