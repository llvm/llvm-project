#include "check.h"

namespace benchmark {
namespace internal {

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
AbortHandlerT* handler = &std::abort;
}  // namespace

BENCHMARK_EXPORT AbortHandlerT*& GetAbortHandler() { return handler; }

}  // namespace internal
}  // namespace benchmark
