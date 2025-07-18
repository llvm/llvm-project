#ifndef MATHTEST_ERRORHANDLING_HPP
#define MATHTEST_ERRORHANDLING_HPP

#include "mathtest/OffloadForward.hpp"

#include "llvm/ADT/Twine.h"

#define FATAL_ERROR(Message)                                                   \
  mathtest::detail::reportFatalError(Message, __FILE__, __LINE__, __func__)

#define OL_CHECK(ResultExpr)                                                   \
  do {                                                                         \
    ol_result_t Result = (ResultExpr);                                         \
    if (Result != OL_SUCCESS) {                                                \
      mathtest::detail::reportOffloadError(#ResultExpr, Result, __FILE__,      \
                                           __LINE__, __func__);                \
    }                                                                          \
  } while (false)

namespace mathtest {
namespace detail {

[[noreturn]] void reportFatalError(const llvm::Twine &Message, const char *File,
                                   int Line, const char *FuncName);

[[noreturn]] void reportOffloadError(const char *ResultExpr, ol_result_t Result,
                                     const char *File, int Line,
                                     const char *FuncName);
} // namespace detail
} // namespace mathtest

#endif // MATHTEST_ERRORHANDLING_HPP
