#ifndef LIBOMPTARGET_PLUGINS_AMGGPU_SRC_TRACE_H_INCLUDED
#define LIBOMPTARGET_PLUGINS_AMGGPU_SRC_TRACE_H_INCLUDED

#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <omptarget.h>
#include <tuple>
#include <utility>

#include "print_tracing.h"

namespace {
namespace detail {

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
  log_t(const char *func, Ts &&... args)
      : func(func), args(std::forward<Ts>(args)...) {
    active = print_kernel_trace & RTL_TIMING;  // is bit 1 set ?

    if (!active) {
      return;
    }

    start = clock_ty::now();
  }

  void res(R r) { result = r; }

  template <size_t... Is>
  int printUnpack(int64_t t, std::tuple<Ts...> const &tup,
                  std::index_sequence<Is...>) {

    return fprintf(print_kernel_trace & RTL_TO_STDOUT ? stdout : stderr,
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
log_t<R, Ts...> log(const char *func, Ts &&... ts) {
  return log_t<R, Ts...>(func, std::forward<Ts>(ts)...);
}

} // namespace detail
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

static void *__tgt_rtl_data_alloc_impl(int device_id, int64_t size, void *ptr, int32_t kind);
void *__tgt_rtl_data_alloc(int device_id, int64_t size, void *ptr, int32_t Kind) {
  auto t = detail::log<void *>(__func__, device_id, size, ptr);
  void *r = __tgt_rtl_data_alloc_impl(device_id, size, ptr, Kind);
  t.res(r);
  return r;
}
#define __tgt_rtl_data_alloc(...) __tgt_rtl_data_alloc_impl(__VA_ARGS__)

static int32_t __tgt_rtl_data_delete_impl(int device_id, void *tgt_ptr);
int32_t __tgt_rtl_data_delete(int device_id, void *tgt_ptr) {
  auto t = detail::log<int32_t>(__func__, device_id, tgt_ptr);
  int32_t r = __tgt_rtl_data_delete_impl(device_id, tgt_ptr);
  t.res(r);
  return r;
}
#define __tgt_rtl_data_delete(...) __tgt_rtl_data_delete_impl(__VA_ARGS__)

static int32_t __tgt_rtl_data_retrieve_impl(int device_id, void *hst_ptr,
                                            void *tgt_ptr, int64_t size);
int32_t __tgt_rtl_data_retrieve(int device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  auto t = detail::log<int32_t>(__func__, device_id, hst_ptr, tgt_ptr, size);
  int32_t r = __tgt_rtl_data_retrieve_impl(device_id, hst_ptr, tgt_ptr, size);
  t.res(r);
  return r;
}
#define __tgt_rtl_data_retrieve(...) __tgt_rtl_data_retrieve_impl(__VA_ARGS__)

static int32_t
__tgt_rtl_data_retrieve_async_impl(int device_id, void *hst_ptr, void *tgt_ptr,
                                   int64_t size,
                                   __tgt_async_info *async_info_ptr);
int32_t __tgt_rtl_data_retrieve_async(int device_id, void *hst_ptr,
                                      void *tgt_ptr, int64_t size,
                                      __tgt_async_info *async_info_ptr) {
  auto t = detail::log<int32_t>(__func__, device_id, hst_ptr, tgt_ptr, size,
                                async_info_ptr);
  int32_t r = __tgt_rtl_data_retrieve_async_impl(device_id, hst_ptr, tgt_ptr,
                                                 size, async_info_ptr);
  t.res(r);
  return r;
}
#define __tgt_rtl_data_retrieve_async(...)                                     \
  __tgt_rtl_data_retrieve_async_impl(__VA_ARGS__)

static int32_t __tgt_rtl_data_submit_impl(int device_id, void *tgt_ptr,
                                          void *hst_ptr, int64_t size);
int32_t __tgt_rtl_data_submit(int device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  auto t = detail::log<int32_t>(__func__, device_id, tgt_ptr, hst_ptr, size);
  int32_t r = __tgt_rtl_data_submit_impl(device_id, tgt_ptr, hst_ptr, size);
  t.res(r);
  return r;
}
#define __tgt_rtl_data_submit(...) __tgt_rtl_data_submit_impl(__VA_ARGS__)

static int32_t __tgt_rtl_data_submit_async_impl(int32_t ID, void *TargetPtr,
                                                void *HostPtr, int64_t Size,
                                                __tgt_async_info *AsyncInfoPtr);
int32_t __tgt_rtl_data_submit_async(int32_t ID, void *TargetPtr, void *HostPtr,
                                    int64_t Size,
                                    __tgt_async_info *AsyncInfoPtr) {
  auto t = detail::log<int32_t>(__func__, ID, TargetPtr, HostPtr, Size,
                                AsyncInfoPtr);
  int32_t r = __tgt_rtl_data_submit_async_impl(ID, TargetPtr, HostPtr, Size,
                                               AsyncInfoPtr);
  t.res(r);
  return r;
}
#define __tgt_rtl_data_submit_async(...)                                       \
  __tgt_rtl_data_submit_async_impl(__VA_ARGS__)

static int32_t __tgt_rtl_init_device_impl(int device_id);
int32_t __tgt_rtl_init_device(int device_id) {
  auto t = detail::log<int32_t>(__func__, device_id);
  int32_t r = __tgt_rtl_init_device_impl(device_id);
  t.res(r);
  return r;
}
#define __tgt_rtl_init_device(...) __tgt_rtl_init_device_impl(__VA_ARGS__)

static int64_t __tgt_rtl_init_requires_impl(int64_t RequiresFlags);
int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
  auto t = detail::log<int64_t>(__func__, RequiresFlags);
  int64_t r = __tgt_rtl_init_requires_impl(RequiresFlags);
  t.res(r);
  return r;
}
#define __tgt_rtl_init_requires(...) __tgt_rtl_init_requires_impl(__VA_ARGS__)

static int32_t __tgt_rtl_is_valid_binary_impl(__tgt_device_image *image);
int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {
  auto t = detail::log<int32_t>(__func__, image);
  int32_t r = __tgt_rtl_is_valid_binary_impl(image);
  t.res(r);
  return r;
}
#define __tgt_rtl_is_valid_binary(...)                                         \
  __tgt_rtl_is_valid_binary_impl(__VA_ARGS__)

static __tgt_target_table *
__tgt_rtl_load_binary_impl(int32_t device_id, __tgt_device_image *image);
__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {
  auto t = detail::log<__tgt_target_table *>(__func__, device_id, image);
  __tgt_target_table *r = __tgt_rtl_load_binary_impl(device_id, image);
  t.res(r);
  return r;
}
#define __tgt_rtl_load_binary(...) __tgt_rtl_load_binary_impl(__VA_ARGS__)

static int __tgt_rtl_number_of_devices_impl();
int __tgt_rtl_number_of_devices() {
  auto t = detail::log<int>(__func__);
  int r = __tgt_rtl_number_of_devices_impl();
  t.res(r);
  return r;
}
#define __tgt_rtl_number_of_devices(...)                                       \
  __tgt_rtl_number_of_devices_impl(__VA_ARGS__)

static int32_t __tgt_rtl_run_target_region_impl(int32_t device_id,
                                                void *tgt_entry_ptr,
                                                void **tgt_args,
                                                ptrdiff_t *tgt_offsets,
                                                int32_t arg_num);
int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets,
                                    int32_t arg_num) {
  auto t = detail::log<int32_t>(__func__, device_id, tgt_entry_ptr, tgt_args,
                                tgt_offsets, arg_num);
  int32_t r = __tgt_rtl_run_target_region_impl(device_id, tgt_entry_ptr,
                                               tgt_args, tgt_offsets, arg_num);
  t.res(r);
  return r;
}
#define __tgt_rtl_run_target_region(...)                                       \
  __tgt_rtl_run_target_region_impl(__VA_ARGS__)

static int32_t __tgt_rtl_run_target_region_async_impl(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, __tgt_async_info *async_info_ptr);
int32_t __tgt_rtl_run_target_region_async(int32_t device_id,
                                          void *tgt_entry_ptr, void **tgt_args,
                                          ptrdiff_t *tgt_offsets,
                                          int32_t arg_num,
                                          __tgt_async_info *async_info_ptr) {
  auto t = detail::log<int32_t>(__func__, device_id, tgt_entry_ptr, tgt_args,
                                tgt_offsets, arg_num, async_info_ptr);
  int32_t r = __tgt_rtl_run_target_region_async_impl(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, async_info_ptr);
  t.res(r);
  return r;
}
#define __tgt_rtl_run_target_region_async(...)                                 \
  __tgt_rtl_run_target_region_async_impl(__VA_ARGS__)

static int32_t __tgt_rtl_run_target_team_region_impl(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t num_teams,
    int32_t thread_limit, uint64_t loop_tripcount);
int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args,
                                         ptrdiff_t *tgt_offsets,
                                         int32_t arg_num, int32_t num_teams,
                                         int32_t thread_limit,
                                         uint64_t loop_tripcount) {
  auto t = detail::log<int32_t>(__func__, device_id, tgt_entry_ptr, tgt_args,
                                tgt_offsets, arg_num, num_teams, thread_limit,
                                loop_tripcount);
  int32_t r = __tgt_rtl_run_target_team_region_impl(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, num_teams,
      thread_limit, loop_tripcount);
  t.res(r);
  return r;
}
#define __tgt_rtl_run_target_team_region(...)                                  \
  __tgt_rtl_run_target_team_region_impl(__VA_ARGS__)

static int32_t __tgt_rtl_run_target_team_region_async_impl(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t num_teams,
    int32_t thread_limit, uint64_t loop_tripcount,
     __tgt_async_info *AsyncInfo);
int32_t __tgt_rtl_run_target_team_region_async(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args,
                                         ptrdiff_t *tgt_offsets,
                                         int32_t arg_num, int32_t num_teams,
                                         int32_t thread_limit,
                                         uint64_t loop_tripcount,
					  __tgt_async_info *AsyncInfo) {
  auto t = detail::log<int32_t>(__func__, device_id, tgt_entry_ptr, tgt_args,
                                tgt_offsets, arg_num, num_teams, thread_limit,
                                loop_tripcount, AsyncInfo);
  int32_t r = __tgt_rtl_run_target_team_region_async_impl(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, num_teams,
      thread_limit, loop_tripcount, AsyncInfo);
  t.res(r);
  return r;
}
#define __tgt_rtl_run_target_team_region_async(...)                                  \
  __tgt_rtl_run_target_team_region_async_impl(__VA_ARGS__)

static int32_t __tgt_rtl_synchronize_impl(int32_t device_id,
                                          __tgt_async_info *async_info_ptr);
int32_t __tgt_rtl_synchronize(int32_t device_id,
                              __tgt_async_info *async_info_ptr) {
  auto t = detail::log<int32_t>(__func__, device_id, async_info_ptr);
  int32_t r = __tgt_rtl_synchronize_impl(device_id, async_info_ptr);
  t.res(r);
  return r;
}
#define __tgt_rtl_synchronize(...) __tgt_rtl_synchronize_impl(__VA_ARGS__)

static int32_t __tgt_rtl_set_coarse_grain_mem_region_impl(void *ptr,
                                                          int64_t size);
int32_t __tgt_rtl_set_coarse_grain_mem_region(void *ptr, int64_t size) {
  auto t = detail::log<int32_t>(__func__, ptr, size);
  int32_t r = __tgt_rtl_set_coarse_grain_mem_region_impl(ptr, size);
  t.res(r);
  return r;
}
#define __tgt_rtl_set_coarse_grain_mem_region(...)                             \
  __tgt_rtl_set_coarse_grain_mem_region_impl(__VA_ARGS__)

static int32_t __tgt_rtl_query_coarse_grain_mem_region_impl(const void *ptr,
                                                            int64_t size);
int32_t __tgt_rtl_query_coarse_grain_mem_region(const void *ptr, int64_t size) {
  auto t = detail::log<int32_t>(__func__, ptr, size);
  int32_t r = __tgt_rtl_query_coarse_grain_mem_region_impl(ptr, size);
  t.res(r);
  return r;
}
#define __tgt_rtl_query_coarse_grain_mem_region(...)                           \
  __tgt_rtl_query_coarse_grain_mem_region_impl(__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif
