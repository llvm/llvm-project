#ifndef KMP_NS_H
#define KMP_NS_H

#include <cstdlib>

#include "kmp.h"
#ifdef LIBOMP_NEXTSILICON
#include "nsapi/intrinsics.h"
#include "nsapi/omp.h"
#endif

extern "C" int __kmp_get_global_thread_id(void);

// do nothing for standard gtid,
// but if gtid < 0, then pickup the gtid from the TLS variable.
inline int32_t __kmp_ns_gtid_restore_if_risc(int32_t gtid) {
#ifdef LIBOMP_NEXTSILICON
  if (gtid < 0)
    return __kmp_get_global_thread_id();
#endif
  return gtid;
}

#ifdef LIBOMP_NEXTSILICON

extern "C" int32_t __kmp_ns_gtid_encode(uint32_t team_size,
                                        uint32_t thread_idx);

/// When OpenMP mode is enabled, this will be transformed into using a size
/// argument passed into the offloaded NextSilicon device kernel. When OpenMP
/// mode is disabled (deprecated), this unpacks the team size that has been
/// packed into the OpenMP GTID.
extern "C" uint32_t __kmp_ns_gtid_decode_num_threads(int32_t gtid);

/// When OpenMP mode is enabled, this will be transformed into a direct lookup
/// of the current thread index. When OpenMP mode is disabled (deprecated), this
/// unpacks the thread index that has been packed into the OpenMP GTID.
extern "C" uint32_t __kmp_ns_gtid_decode_thread_index(int32_t gtid);

extern "C" int32_t
__kmp_ns_gtid_encode_from_gtid_tid_and_size(int32_t gtid, int32_t tid,
                                            uint32_t team_size);

/**
 * Returns the OpenMP thread index of the current thread
 *
 * @orig_tid: original thread index (`tid`) argument passed to the microtask
 * @return: `orig_tid` when running on host, thread index in NextSilicon team
 * when running on device
 */
extern "C" kmp_int32 __kmp_ns_get_tid(kmp_int32 *orig_tid);

/**
 * Returns the OpenMP unique thread ID (GTID) of the current thread
 *
 * @orig_gtid: original unique thread ID (`gtid`) argument passed to the
 * microtask
 * @thread_idx: thread index in the team, as returned by `__kmp_ns_get_tid`
 * @return: `orig_gtid` when running on host, crafted NextSilicon
 * device-specific unique thread ID when running on device NOTE: crafted thread
 * ID has the thread index encoded in it (as well as team base and size)
 */
extern "C" kmp_int32 __kmp_ns_get_gtid(kmp_int32 *orig_gtid,
                                       kmp_int32 thread_idx);

#ifdef LIBOMP_NEXTSILICON_ATOMICS_BYPASS

extern int __kmp_ns_atomics_bypass_enable;

/// Special handling for malloc/free for OpenMP memory
extern "C" void *__kmp_ns_malloc(size_t size);
extern "C" void *__kmp_ns_calloc(size_t num, size_t size);
extern "C" void *__kmp_ns_realloc(void *ptr, size_t new_size);
extern "C" void __kmp_ns_free(void *addr);

#endif // LIBOMP_NEXTSILICON_ATOMICS_BYPASS

#endif // LIBOMP_NEXTSILICON

#endif // KMP_NS_H
