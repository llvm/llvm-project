/*
 * kmp_ns.cpp -- NextSilicon OpenMP extension
 */
#include "kmp_ns.h"

#include <cstdint>

#ifdef LIBOMP_NEXTSILICON_ATOMICS_BYPASS
#include "tinyalloc/tinyalloc_wrapper.hpp"
#endif

enum class kmp_ns_openmp_origin : int { risc = 0, host = 1 };

// "negative GTID" usage and bit fields:
// Supporting small footprint RISC OpenMP, provisions for simpler static sched
// via the gtid as a bit fields is defined.
// Since static scheduling depends only on the team size and particular member
// within a team, the gtid is encoded (and decoded) in a way which the RISC
// OpenMP would call the parallel region, namely, the 'omp_outlined' gtid
// argument.
//
// bit   31         "1" - indicates negative GTID, such that team size and tid
//                        are encoded in the gtid itself.
//                  "0" - standard host openmp gtid, which is an index to the
//                  __kmp_threads[]
//                        table, where the tid and team size are picked up in
//                        the normal way.
// bit   30         "1" - indicates host initiated this parallel region.
//                  "0" - indicates this parallel region was started on the
//                        risc.
// bits  29:28      Unused
// bits  27:14      team size
// bits  13:0       tid = member within the team size, such that  0 <= tid <
// team_size
//
// Notes:
// 1. negative gtid mechanism supports teams of up to 16,384 - 1 threads.
// 2. there are two bits encoding host/risc, for the following reason:
//    a. bit[31] indicates that this is a negative gtid, and thus, team size and
//    member tid
//       are encoded here. However, this can also be used by the host, if the
//       option is enabled and the team size is small enough.
//    b. bit[30] actually indicates risc/host, since there are additional
//    reasons to distinguish
//       host/risc, which the team size/member tid are not sufficient.
//       the reasoning is that special hypercalls which are called by both host
//       and risc openmp need to distinguish the two environments.
constexpr uint32_t KMP_NS_SCHED_GTID_TEAM_SIZE_BIT_OFFSET = 14;
constexpr uint32_t KMP_NS_SCHED_GTID_TEAM_SIZE_BIT_SIZE = 14;
constexpr uint32_t KMP_NS_SCHED_GTID_TID_BIT_OFFSET = 0;
constexpr uint32_t KMP_NS_SCHED_GTID_TID_BIT_SIZE = 14;
constexpr uint32_t KMP_NS_SCHED_GTID_ORIGIN_BIT = 30;
constexpr uint32_t KMP_NS_SCHED_GTID_RISC_BIT = 31;

constexpr uint32_t KMP_NS_SCHED_GTID_MAX_TEAM_SIZE =
    (1UL << KMP_NS_SCHED_GTID_TEAM_SIZE_BIT_SIZE) - 1;

extern "C" int32_t __kmp_ns_gtid_encode(uint32_t num_threads,
                                        uint32_t thread_index) {

  uint32_t gtid_u32 = (num_threads << KMP_NS_SCHED_GTID_TEAM_SIZE_BIT_OFFSET) |
                      (thread_index << KMP_NS_SCHED_GTID_TID_BIT_OFFSET) |
                      (1UL << KMP_NS_SCHED_GTID_RISC_BIT);

  return static_cast<int32_t>(gtid_u32);
}

// Bits.......11111111.........
//            y      x
template <typename T>
constexpr T __kmp_ns_gtid_generate_bitmask(uint32_t upper_bit_index,
                                           uint32_t lower_bit_index) {
  if (upper_bit_index < lower_bit_index)
    return static_cast<T>(0);

  if (lower_bit_index == 0)
    return (static_cast<T>(1) << (upper_bit_index + 1)) - 1;

  return (static_cast<T>(1) << (upper_bit_index + 1)) - 1 -
         ((static_cast<T>(1) << (lower_bit_index + 1)) - 1);
}

extern "C" uint32_t __kmp_ns_gtid_decode_num_threads(int32_t gtid) {
  // This is a workaround implementation required to run OpenMP in the
  // NextSilicon system without fork point lifting. Instead of having the number
  // of threads provided passed in as a kernel argument, we unpack it from the
  // OpenMP GTID. This should be removed once it's no longer possible to disable
  // NextSilicon OpenMP mode.
  uint32_t gtid_u32 = static_cast<uint32_t>(gtid);

  return (gtid_u32 >> KMP_NS_SCHED_GTID_TEAM_SIZE_BIT_OFFSET) &
         __kmp_ns_gtid_generate_bitmask<uint32_t>(
             KMP_NS_SCHED_GTID_TEAM_SIZE_BIT_SIZE - 1, 0);
}

extern "C" uint32_t __kmp_ns_gtid_decode_thread_index(int32_t gtid) {
  // This is a workaround implementation required to run OpenMP in the
  // NextSilicon system without fork point lifting. Instead of having the thread
  // index provided directly by the hardware, we unpack it from the OpenMP GTID.
  // This should be removed once it's no longer possible to disable NextSilicon
  // OpenMP mode.
  uint32_t gtid_u32 = static_cast<uint32_t>(gtid);

  return (gtid_u32 >> KMP_NS_SCHED_GTID_TID_BIT_OFFSET) &
         __kmp_ns_gtid_generate_bitmask<uint32_t>(
             KMP_NS_SCHED_GTID_TID_BIT_SIZE - 1, 0);
}
extern "C" int32_t
__kmp_ns_gtid_encode_from_gtid_tid_and_size(int32_t gtid, int32_t tid,
                                            uint32_t num_threads) {
  if (num_threads > KMP_NS_SCHED_GTID_MAX_TEAM_SIZE)
    return gtid;

  uint32_t utid = static_cast<uint32_t>(tid);

  uint32_t gtid_u32 = (num_threads << KMP_NS_SCHED_GTID_TEAM_SIZE_BIT_OFFSET) |
                      (utid << KMP_NS_SCHED_GTID_TID_BIT_OFFSET) |
                      (static_cast<uint32_t>(kmp_ns_openmp_origin::host)
                       << KMP_NS_SCHED_GTID_ORIGIN_BIT) |
                      (1UL << KMP_NS_SCHED_GTID_RISC_BIT);

  return static_cast<int32_t>(gtid_u32);
}

extern "C" kmp_int32 __kmp_ns_get_tid(kmp_int32 *orig_tid) {
  // Check both that we are handed off (on CG)
  // AND we came from __ns_team_spawn (so the TID/GTID arguments are nulls)
  if (__nsapi_is_on_cg() && orig_tid == nullptr) {
    return __nsapi_team_get_thread_index();
  } else {
    return *orig_tid;
  }
}

extern "C" kmp_int32 __kmp_ns_get_gtid(kmp_int32 *orig_gtid,
                                       kmp_int32 thread_idx) {
  // Check both that we are handed off (on CG)
  // AND we came from __ns_team_spawn (so the TID/GTID arguments are nulls)
  if (__nsapi_is_on_cg() && orig_gtid == nullptr) {
    return __kmp_ns_gtid_encode(__nsapi_team_get_team_size(),
                                static_cast<uint32_t>(thread_idx));
  } else {
    return *orig_gtid;
  }
}

#ifdef LIBOMP_NEXTSILICON_ATOMICS_BYPASS

int __kmp_ns_atomics_bypass_enable = true;

/// Temporary aorkaround for SOF-6554
/// Enforce all OpenMP system allocations to be marked,
/// in the NextSilicon way, as non-migratable memory.
/// This is to avoid from OpenMP runtime code to use atomic
/// instructions over memory migrated behind the PCI.
extern "C" void *__kmp_ns_malloc(size_t size) {
  return ns::tinyalloc::wrapper::alloc(size);
}

extern "C" void __kmp_ns_free(void *addr) {
  ns::tinyalloc::wrapper::free(addr);
}

extern "C" void *__kmp_ns_calloc(size_t num, size_t size) {
  return ns::tinyalloc::wrapper::calloc(num, size);
}

extern "C" void *__kmp_ns_realloc(void *ptr, size_t new_size) {
  return ns::tinyalloc::wrapper::realloc(ptr, new_size);
}

#endif // LIBOMP_NEXTSILICON_ATOMICS_BYPASS
