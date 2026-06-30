//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CHRONO_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CHRONO_H

#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

namespace lldb_private::formatters::chrono {

#pragma region Calendar

/// std::chrono::day
bool DaySummaryProvider(ValueObject &valobj, Stream &stream,
                        const TypeSummaryOptions &options);

/// std::chrono::month
bool MonthSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options);

/// std::chrono::year
bool YearSummaryProvider(ValueObject &valobj, Stream &stream,
                         const TypeSummaryOptions &options);

/// std::chrono::weekday
bool WeekdaySummaryProvider(ValueObject &valobj, Stream &stream,
                            const TypeSummaryOptions &options);

/// std::chrono::weekday_indexed
bool WeekdayIndexedSummaryProvider(ValueObject &valobj, Stream &stream,
                                   const TypeSummaryOptions &options);

/// std::chrono::weekday_last
bool WeekdayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &options);

/// std::chrono::month_day
bool MonthDaySummaryProvider(ValueObject &valobj, Stream &stream,
                             const TypeSummaryOptions &options);

/// std::chrono::month_day_last
bool MonthDayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options);

/// std::chrono::month_weekday
bool MonthWeekdaySummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options);

/// std::chrono::month_weekday_last
bool MonthWeekdayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                     const TypeSummaryOptions &options);

/// std::chrono::year_month
bool YearMonthSummaryProvider(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &options);

/// std::chrono::year_month_day
bool YearMonthDaySummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options);

/// std::chrono::year_month_day_last
bool YearMonthDayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                     const TypeSummaryOptions &options);

/// std::chrono::year_month_weekday
bool YearMonthWeekdaySummaryProvider(ValueObject &valobj, Stream &stream,
                                     const TypeSummaryOptions &options);

/// std::chrono::year_month_weekday_last
bool YearMonthWeekdayLastSummaryProvider(ValueObject &valobj, Stream &stream,
                                         const TypeSummaryOptions &options);

#pragma region Durations

/// std::chrono::nanoseconds
bool NanosecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &options);

/// std::chrono::microseconds
bool MicrosecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options);

/// std::chrono::milliseconds
bool MillisecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options);

/// std::chrono::seconds
bool SecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                            const TypeSummaryOptions &options);

/// std::chrono::minutes
bool MinutesSummaryProvider(ValueObject &valobj, Stream &stream,
                            const TypeSummaryOptions &options);

/// std::chrono::hours
bool HoursSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options);

/// std::chrono::days
bool DaysSummaryProvider(ValueObject &valobj, Stream &stream,
                         const TypeSummaryOptions &options);

/// std::chrono::weeks
bool WeeksSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options);

/// std::chrono::years
bool YearsSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options);

/// std::chrono::months
bool MonthsSummaryProvider(ValueObject &valobj, Stream &stream,
                           const TypeSummaryOptions &options);

#pragma region Timepoints

/// std::chrono::time_point<system_clock, seconds>
bool SysSecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                               const TypeSummaryOptions &options);

/// std::chrono::time_point<system_clock, days>
bool SysDaysSummaryProvider(ValueObject &valobj, Stream &stream,
                            const TypeSummaryOptions &options);

/// std::chrono::time_point<local_t, seconds>
bool LocalSecondsSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &options);

/// std::chrono::time_point<local_t, days>
bool LocalDaysSummaryProvider(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &options);

} // namespace lldb_private::formatters::chrono

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CHRONO_H
