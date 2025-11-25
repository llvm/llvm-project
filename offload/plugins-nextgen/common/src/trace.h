#ifndef LIBOMPTARGET_PLUGINS_AMGGPU_SRC_TRACE_H_INCLUDED
#define LIBOMPTARGET_PLUGINS_AMGGPU_SRC_TRACE_H_INCLUDED

#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <omptarget.h>
#include <tuple>
#include <utility>

#include "Shared/Debug.h"

namespace {

namespace logger {

// Plumbing for concatenating format strings
template <size_t N, size_t... Is>
constexpr std::array<const char, N - 1> toArray(const char (&a)[N],
                                                std::index_sequence<Is...>) {
  return {{a[Is]...}};
}
template <size_t N, size_t... Is>
constexpr std::array<const char, N - 1> toArray(const char (&a)[N]) {
  return toArray(a, std::make_index_sequence<N - 1>());
}
template <size_t N, size_t M, size_t... lhs_is, size_t... rhs_is>
constexpr std::array<const char, N + M>
cat(std::array<const char, N> const &lhs, std::array<const char, M> const &rhs,
    std::index_sequence<lhs_is...>, std::index_sequence<rhs_is...>) {
  return {lhs[lhs_is]..., rhs[rhs_is]...};
}
template <size_t N, size_t M>
constexpr std::array<const char, N + M>
cat(std::array<const char, N> const &lhs,
    std::array<const char, M> const &rhs) {
  return cat(lhs, rhs, std::make_index_sequence<N>(),
             std::make_index_sequence<M>());
}
template <size_t N, size_t M, size_t O>
constexpr std::array<const char, N + M + O>
cat(std::array<const char, N> const &x, std::array<const char, M> const &y,
    std::array<const char, O> const &z) {
  return cat(cat(x, y), z);
}

// Print pointers as 48 bit hex, integers as same width
template <typename T> struct fmt;
template <> struct fmt<bool> {
  static constexpr auto value() { return toArray("%14" PRId32); }
};
template <> struct fmt<int32_t> {
  static constexpr auto value() { return toArray("%14" PRId32); }
};
template <> struct fmt<int64_t> {
  static constexpr auto value() { return toArray("%14" PRId64); }
};
template <> struct fmt<uint64_t> {
  static constexpr auto value() { return toArray("%14" PRIu64); }
};
template <> struct fmt<void *> {
  static constexpr auto value() { return toArray("0x%.12" PRIxPTR); }
};
template <typename T> struct fmt<T *> {
  static constexpr auto value() { return fmt<void *>::value(); }
};

// Format function arguments as 'function:   time us (x, y, z)'
template <size_t I> struct delimiter {
  static constexpr auto value() { return toArray(", "); }
};

template <> struct delimiter<0> {
  static constexpr auto value() { return toArray("("); }
};

template <size_t I, typename... Ts,
          typename std::enable_if<I == sizeof...(Ts), int>::type = 0>
constexpr std::array<const char, 1> fmtTupleFrom() {
  return toArray(")");
}

template <size_t I, typename... Ts,
          typename std::enable_if<I<sizeof...(Ts), int>::type =
                                      0> constexpr auto fmtTupleFrom() {
  using type = typename std::tuple_element<I, std::tuple<Ts...>>::type;
  constexpr auto f = fmt<typename std::decay<type>::type>::value();
  constexpr auto r = fmtTupleFrom<I + 1, Ts...>();
  return cat(delimiter<I>::value(), f, r);
}

template <typename... Ts> constexpr auto fmtTuple() {
  return fmtTupleFrom<0, Ts...>();
}

// This composes the format string at compile time without putting a copy on the
// stack. C++14 requires an out of line declaration for static variables, and
// c++ requires an initializer for auto variables. C++ rejects an initializer on
// the declaration so the type must be explicit. In this case, it is dependent
// on Ts, get() and exposing size() work around. GCC has a bug where it fails to
// recognise that ::value defined using size() and using fmtStr<Ts...>::size()
// are the same type, worked around using the longer spelling.
// Writing the contents of fmtStr::get() inline in log_t is simpler, but puts
// a ~100 byte object on the stack and calls memcpy on it.
template <typename R, typename... Ts> class fmtStr {
  static constexpr auto get() {
    // Call function: 123us result (some, number, of, arguments)
    return cat(cat(toArray("Call %35s: %8" PRId64 "us "),
                   fmt<typename std::decay<R>::type>::value(), toArray(" ")),
               cat(fmtTuple<Ts...>(), toArray("\n\0")));
  }

public:
  static constexpr size_t size() { return get().size(); }
  static constexpr const std::array<const char, fmtStr<R, Ts...>::size()>
      value = get();
  static constexpr const char *data() { return value.data(); }
};
template <typename R, typename... Ts>
constexpr const std::array<const char, fmtStr<R, Ts...>::size()>
    fmtStr<R, Ts...>::value;

template <typename R, typename... Ts> struct log_t {
  using clock_ty = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock_ty> start, end;

  const char *func;
  std::tuple<Ts...> args;
  bool active;
  R result;
  log_t(const char *func, Ts &&...args)
      : func(func), args(std::forward<Ts>(args)...) {
    active = getInfoLevel() & OMP_INFOTYPE_AMD_API_TRACE;

    if (!active) {
      return;
    }

    start = clock_ty::now();
  }

  void res(R r) { result = r; }

  template <size_t... Is>
  int printUnpack(int64_t t, std::tuple<Ts...> const &tup,
                  std::index_sequence<Is...>) {

    return fprintf(getInfoLevel() & RTL_TO_STDOUT ? stdout : stderr,
                   fmtStr<R, Ts...>::data(), func, t, result,
                   std::get<Is>(tup)...);
  }

  ~log_t() {
    if (!active) {
      return;
    }
    end = clock_ty::now();
    int64_t t =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    printUnpack(t, args, std::make_index_sequence<sizeof...(Ts)>());
  }
};

template <typename R, typename... Ts>
log_t<R, Ts...> log(const char *func, Ts &&...ts) {
  return log_t<R, Ts...>(func, std::forward<Ts>(ts)...);
}

} // namespace logger
} // namespace

#endif
