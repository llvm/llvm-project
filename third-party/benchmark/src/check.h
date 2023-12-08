#ifndef CHECK_H_
#define CHECK_H_

#include <cmath>
#include <cstdlib>
#include <ostream>

#include "internal_macros.h"
#include "log.h"

namespace benchmark {
namespace internal {

typedef void(AbortHandlerT)();

inline AbortHandlerT*& GetAbortHandler() {
  static AbortHandlerT* handler = &std::abort;
  return handler;
}

BENCHMARK_NORETURN inline void CallAbortHandler() {
  GetAbortHandler()();
  std::abort();  // fallback to enforce noreturn
}

// CheckHandler is the class constructed by failing BM_CHECK macros.
// CheckHandler will log information about the failures and abort when it is
// destructed.
class CheckHandler {
 public:
  CheckHandler(const char* check, const char* file, const char* func, int line)
      : log_(GetErrorLogInstance()) {
    log_ << file << ":" << line << ": " << func << ": Check `" << check
         << "' failed. ";
  }

  LogType& GetLog() { return log_; }

  BENCHMARK_NORETURN ~CheckHandler() BENCHMARK_NOEXCEPT_OP(false) {
    log_ << std::endl;
    CallAbortHandler();
  }

  CheckHandler& operator=(const CheckHandler&) = delete;
  CheckHandler(const CheckHandler&) = delete;
  CheckHandler() = delete;

 private:
  LogType& log_;
};

}  // end namespace internal
}  // end namespace benchmark

// The BM_CHECK macro returns a std::ostream object that can have extra
// information written to it.
#ifndef NDEBUG
#define BM_CHECK(b)                                                          \
  (b ? ::benchmark::internal::GetNullLogInstance()                           \
     : ::benchmark::internal::CheckHandler(#b, __FILE__, __func__, __LINE__) \
           .GetLog())
#else
#define BM_CHECK(b) ::benchmark::internal::GetNullLogInstance()
#endif

// clang-format off
// preserve whitespacing between operators for alignment
#define BM_CHECK_EQ(a, b) BM_CHECK((a) == (b))
#define BM_CHECK_NE(a, b) BM_CHECK((a) != (b))
#define BM_CHECK_GE(a, b) BM_CHECK((a) >= (b))
#define BM_CHECK_LE(a, b) BM_CHECK((a) <= (b))
#define BM_CHECK_GT(a, b) BM_CHECK((a) > (b))
#define BM_CHECK_LT(a, b) BM_CHECK((a) < (b))

#define BM_CHECK_FLOAT_EQ(a, b, eps) BM_CHECK(std::fabs((a) - (b)) <  (eps))
#define BM_CHECK_FLOAT_NE(a, b, eps) BM_CHECK(std::fabs((a) - (b)) >= (eps))
#define BM_CHECK_FLOAT_GE(a, b, eps) BM_CHECK((a) - (b) > -(eps))
#define BM_CHECK_FLOAT_LE(a, b, eps) BM_CHECK((b) - (a) > -(eps))
#define BM_CHECK_FLOAT_GT(a, b, eps) BM_CHECK((a) - (b) >  (eps))
#define BM_CHECK_FLOAT_LT(a, b, eps) BM_CHECK((b) - (a) >  (eps))
//clang-format on

#endif  // CHECK_H_
