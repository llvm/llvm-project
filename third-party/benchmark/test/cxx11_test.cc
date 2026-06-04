#include "benchmark/benchmark.h"

#if defined(_MSC_VER)
#if _MSVC_LANG != 201402L
// MSVC, even in C++11 mode, dooes not claim to be in C++11 mode.
#error "Trying to compile C++11 test with wrong C++ standard"
#endif  //  _MSVC_LANG
#else   // Non-MSVC
#if __cplusplus != 201103L
#error "Trying to compile C++11 test with wrong C++ standard"
#endif  // Non-MSVC
#endif
