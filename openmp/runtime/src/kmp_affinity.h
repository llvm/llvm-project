/*
 * kmp_affinity.h -- header for affinity management
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_AFFINITY_H
#define KMP_AFFINITY_H

#include "kmp.h"
#include "kmp_os.h"
#include <limits>

#if KMP_AFFINITY_SUPPORTED
#if KMP_USE_HWLOC
class KMPHwlocAffinity : public KMPAffinity {
public:
  class Mask : public KMPAffinity::Mask {
    hwloc_cpuset_t mask;

  public:
    Mask() {
      mask = hwloc_bitmap_alloc();
      this->zero();
    }
    ~Mask() { hwloc_bitmap_free(mask); }
    void set(int i) override { hwloc_bitmap_set(mask, i); }
    bool is_set(int i) const override { return hwloc_bitmap_isset(mask, i); }
    void clear(int i) override { hwloc_bitmap_clr(mask, i); }
    void zero() override { hwloc_bitmap_zero(mask); }
    bool empty() const override { return hwloc_bitmap_iszero(mask); }
    void copy(const KMPAffinity::Mask *src) override {
      const Mask *convert = static_cast<const Mask *>(src);
      hwloc_bitmap_copy(mask, convert->mask);
    }
    void bitwise_and(const KMPAffinity::Mask *rhs) override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      hwloc_bitmap_and(mask, mask, convert->mask);
    }
    void bitwise_or(const KMPAffinity::Mask *rhs) override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      hwloc_bitmap_or(mask, mask, convert->mask);
    }
    void bitwise_not() override { hwloc_bitmap_not(mask, mask); }
    bool is_equal(const KMPAffinity::Mask *rhs) const override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      return hwloc_bitmap_isequal(mask, convert->mask);
    }
    int begin() const override { return hwloc_bitmap_first(mask); }
    int end() const override { return -1; }
    int next(int previous) const override {
      return hwloc_bitmap_next(mask, previous);
    }
    int get_system_affinity(bool abort_on_error) override {
      KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),
                  "Illegal get affinity operation when not capable");
      long retval =
          hwloc_get_cpubind(__kmp_hwloc_topology, mask, HWLOC_CPUBIND_THREAD);
      if (retval >= 0) {
        return 0;
      }
      int error = errno;
      if (abort_on_error) {
        __kmp_fatal(KMP_MSG(FunctionError, "hwloc_get_cpubind()"),
                    KMP_ERR(error), __kmp_msg_null);
      }
      return error;
    }
    int set_system_affinity(bool abort_on_error) const override {
      KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),
                  "Illegal set affinity operation when not capable");
      long retval =
          hwloc_set_cpubind(__kmp_hwloc_topology, mask, HWLOC_CPUBIND_THREAD);
      if (retval >= 0) {
        return 0;
      }
      int error = errno;
      if (abort_on_error) {
        __kmp_fatal(KMP_MSG(FunctionError, "hwloc_set_cpubind()"),
                    KMP_ERR(error), __kmp_msg_null);
      }
      return error;
    }
#if KMP_OS_WINDOWS
    int set_process_affinity(bool abort_on_error) const override {
      KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),
                  "Illegal set process affinity operation when not capable");
      int error = 0;
      const hwloc_topology_support *support =
          hwloc_topology_get_support(__kmp_hwloc_topology);
      if (support->cpubind->set_proc_cpubind) {
        int retval;
        retval = hwloc_set_cpubind(__kmp_hwloc_topology, mask,
                                   HWLOC_CPUBIND_PROCESS);
        if (retval >= 0)
          return 0;
        error = errno;
        if (abort_on_error)
          __kmp_fatal(KMP_MSG(FunctionError, "hwloc_set_cpubind()"),
                      KMP_ERR(error), __kmp_msg_null);
      }
      return error;
    }
#endif
    int get_proc_group() const override {
      int group = -1;
#if KMP_OS_WINDOWS
      if (__kmp_num_proc_groups == 1) {
        return 1;
      }
      for (int i = 0; i < __kmp_num_proc_groups; i++) {
        // On windows, the long type is always 32 bits
        unsigned long first_32_bits = hwloc_bitmap_to_ith_ulong(mask, i * 2);
        unsigned long second_32_bits =
            hwloc_bitmap_to_ith_ulong(mask, i * 2 + 1);
        if (first_32_bits == 0 && second_32_bits == 0) {
          continue;
        }
        if (group >= 0) {
          return -1;
        }
        group = i;
      }
#endif /* KMP_OS_WINDOWS */
      return group;
    }
  };
  void determine_capable(const char *var) override {
    const hwloc_topology_support *topology_support;
    if (__kmp_hwloc_topology == NULL) {
      if (hwloc_topology_init(&__kmp_hwloc_topology) < 0) {
        __kmp_hwloc_error = TRUE;
        if (__kmp_affinity.flags.verbose) {
          KMP_WARNING(AffHwlocErrorOccurred, var, "hwloc_topology_init()");
        }
      }
      if (hwloc_topology_load(__kmp_hwloc_topology) < 0) {
        __kmp_hwloc_error = TRUE;
        if (__kmp_affinity.flags.verbose) {
          KMP_WARNING(AffHwlocErrorOccurred, var, "hwloc_topology_load()");
        }
      }
    }
    topology_support = hwloc_topology_get_support(__kmp_hwloc_topology);
    // Is the system capable of setting/getting this thread's affinity?
    // Also, is topology discovery possible? (pu indicates ability to discover
    // processing units). And finally, were there no errors when calling any
    // hwloc_* API functions?
    if (topology_support && topology_support->cpubind->set_thisthread_cpubind &&
        topology_support->cpubind->get_thisthread_cpubind &&
        topology_support->discovery->pu && !__kmp_hwloc_error) {
      // enables affinity according to KMP_AFFINITY_CAPABLE() macro
      KMP_AFFINITY_ENABLE(TRUE);
    } else {
      // indicate that hwloc didn't work and disable affinity
      __kmp_hwloc_error = TRUE;
      KMP_AFFINITY_DISABLE();
    }
  }
  void bind_thread(int which) override {
    KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),
                "Illegal set affinity operation when not capable");
    KMPAffinity::Mask *mask;
    KMP_CPU_ALLOC_ON_STACK(mask);
    KMP_CPU_ZERO(mask);
    KMP_CPU_SET(which, mask);
    __kmp_set_system_affinity(mask, TRUE);
    KMP_CPU_FREE_FROM_STACK(mask);
  }
  KMPAffinity::Mask *allocate_mask() override { return new Mask(); }
  void deallocate_mask(KMPAffinity::Mask *m) override { delete m; }
  KMPAffinity::Mask *allocate_mask_array(int num) override {
    return new Mask[num];
  }
  void deallocate_mask_array(KMPAffinity::Mask *array) override {
    Mask *hwloc_array = static_cast<Mask *>(array);
    delete[] hwloc_array;
  }
  KMPAffinity::Mask *index_mask_array(KMPAffinity::Mask *array,
                                      int index) override {
    Mask *hwloc_array = static_cast<Mask *>(array);
    return &(hwloc_array[index]);
  }
  api_type get_api_type() const override { return HWLOC; }
};
#endif /* KMP_USE_HWLOC */

#if KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_NETBSD || KMP_OS_DRAGONFLY ||     \
    KMP_OS_AIX
#if KMP_OS_LINUX
/* On some of the older OS's that we build on, these constants aren't present
   in <asm/unistd.h> #included from <sys.syscall.h>. They must be the same on
   all systems of the same arch where they are defined, and they cannot change.
   stone forever. */
#include <sys/syscall.h>
#if KMP_ARCH_X86 || KMP_ARCH_ARM
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 241
#elif __NR_sched_setaffinity != 241
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 242
#elif __NR_sched_getaffinity != 242
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_AARCH64
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 122
#elif __NR_sched_setaffinity != 122
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 123
#elif __NR_sched_getaffinity != 123
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_X86_64
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 203
#elif __NR_sched_setaffinity != 203
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 204
#elif __NR_sched_getaffinity != 204
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_PPC64
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 222
#elif __NR_sched_setaffinity != 222
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 223
#elif __NR_sched_getaffinity != 223
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_MIPS
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 4239
#elif __NR_sched_setaffinity != 4239
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 4240
#elif __NR_sched_getaffinity != 4240
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_MIPS64
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 5195
#elif __NR_sched_setaffinity != 5195
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 5196
#elif __NR_sched_getaffinity != 5196
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_LOONGARCH64
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 122
#elif __NR_sched_setaffinity != 122
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 123
#elif __NR_sched_getaffinity != 123
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_RISCV64
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 122
#elif __NR_sched_setaffinity != 122
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 123
#elif __NR_sched_getaffinity != 123
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_VE
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 203
#elif __NR_sched_setaffinity != 203
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 204
#elif __NR_sched_getaffinity != 204
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#elif KMP_ARCH_S390X
#ifndef __NR_sched_setaffinity
#define __NR_sched_setaffinity 239
#elif __NR_sched_setaffinity != 239
#error Wrong code for setaffinity system call.
#endif /* __NR_sched_setaffinity */
#ifndef __NR_sched_getaffinity
#define __NR_sched_getaffinity 240
#elif __NR_sched_getaffinity != 240
#error Wrong code for getaffinity system call.
#endif /* __NR_sched_getaffinity */
#else
#error Unknown or unsupported architecture
#endif /* KMP_ARCH_* */
#elif KMP_OS_FREEBSD || KMP_OS_DRAGONFLY
#include <pthread.h>
#include <pthread_np.h>
#elif KMP_OS_NETBSD
#include <pthread.h>
#include <sched.h>
#elif KMP_OS_AIX
#include <sys/dr.h>
#include <sys/rset.h>
#define VMI_MAXRADS 64 // Maximum number of RADs allowed by AIX.
#endif
class KMPNativeAffinity : public KMPAffinity {
  class Mask : public KMPAffinity::Mask {
    typedef unsigned long mask_t;
    typedef decltype(__kmp_affin_mask_size) mask_size_type;
    static const unsigned int BITS_PER_MASK_T = sizeof(mask_t) * CHAR_BIT;
    static const mask_t ONE = 1;
    mask_size_type get_num_mask_types() const {
      return __kmp_affin_mask_size / sizeof(mask_t);
    }

  public:
    mask_t *mask;
    Mask() { mask = (mask_t *)__kmp_allocate(__kmp_affin_mask_size); }
    ~Mask() {
      if (mask)
        __kmp_free(mask);
    }
    void set(int i) override {
      mask[i / BITS_PER_MASK_T] |= (ONE << (i % BITS_PER_MASK_T));
    }
    bool is_set(int i) const override {
      return (mask[i / BITS_PER_MASK_T] & (ONE << (i % BITS_PER_MASK_T)));
    }
    void clear(int i) override {
      mask[i / BITS_PER_MASK_T] &= ~(ONE << (i % BITS_PER_MASK_T));
    }
    void zero() override {
      mask_size_type e = get_num_mask_types();
      for (mask_size_type i = 0; i < e; ++i)
        mask[i] = (mask_t)0;
    }
    bool empty() const override {
      mask_size_type e = get_num_mask_types();
      for (mask_size_type i = 0; i < e; ++i)
        if (mask[i] != (mask_t)0)
          return false;
      return true;
    }
    void copy(const KMPAffinity::Mask *src) override {
      const Mask *convert = static_cast<const Mask *>(src);
      mask_size_type e = get_num_mask_types();
      for (mask_size_type i = 0; i < e; ++i)
        mask[i] = convert->mask[i];
    }
    void bitwise_and(const KMPAffinity::Mask *rhs) override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      mask_size_type e = get_num_mask_types();
      for (mask_size_type i = 0; i < e; ++i)
        mask[i] &= convert->mask[i];
    }
    void bitwise_or(const KMPAffinity::Mask *rhs) override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      mask_size_type e = get_num_mask_types();
      for (mask_size_type i = 0; i < e; ++i)
        mask[i] |= convert->mask[i];
    }
    void bitwise_not() override {
      mask_size_type e = get_num_mask_types();
      for (mask_size_type i = 0; i < e; ++i)
        mask[i] = ~(mask[i]);
    }
    bool is_equal(const KMPAffinity::Mask *rhs) const override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      mask_size_type e = get_num_mask_types();
      for (mask_size_type i = 0; i < e; ++i)
        if (mask[i] != convert->mask[i])
          return false;
      return true;
    }
    int begin() const override {
      int retval = 0;
      while (retval < end() && !is_set(retval))
        ++retval;
      return retval;
    }
    int end() const override {
      int e;
      __kmp_type_convert(get_num_mask_types() * BITS_PER_MASK_T, &e);
      return e;
    }
    int next(int previous) const override {
      int retval = previous + 1;
      while (retval < end() && !is_set(retval))
        ++retval;
      return retval;
    }
#ifndef KMP_OS_AIX
    int get_system_affinity(bool abort_on_error) override {
      KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),
                  "Illegal get affinity operation when not capable");
#if KMP_OS_LINUX
      long retval =
          syscall(__NR_sched_getaffinity, 0, __kmp_affin_mask_size, mask);
#elif KMP_OS_FREEBSD || KMP_OS_NETBSD || KMP_OS_DRAGONFLY
      int r = pthread_getaffinity_np(pthread_self(), __kmp_affin_mask_size,
                                     reinterpret_cast<cpuset_t *>(mask));
      int retval = (r == 0 ? 0 : -1);
#endif
      if (retval >= 0) {
        return 0;
      }
      int error = errno;
      if (abort_on_error) {
        __kmp_fatal(KMP_MSG(FunctionError, "pthread_getaffinity_np()"),
                    KMP_ERR(error), __kmp_msg_null);
      }
      return error;
    }
    int set_system_affinity(bool abort_on_error) const override {
      KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),
                  "Illegal set affinity operation when not capable");
#if KMP_OS_LINUX
      long retval =
          syscall(__NR_sched_setaffinity, 0, __kmp_affin_mask_size, mask);
#elif KMP_OS_FREEBSD || KMP_OS_NETBSD || KMP_OS_DRAGONFLY
      int r = pthread_setaffinity_np(pthread_self(), __kmp_affin_mask_size,
                                     reinterpret_cast<cpuset_t *>(mask));
      int retval = (r == 0 ? 0 : -1);
#endif
      if (retval >= 0) {
        return 0;
      }
      int error = errno;
      if (abort_on_error) {
        __kmp_fatal(KMP_MSG(FunctionError, "pthread_setaffinity_np()"),
                    KMP_ERR(error), __kmp_msg_null);
      }
      return error;
    }
#elif KMP_OS_AIX
    // On AIX, we don't have a way to get CPU(s) a thread is bound to.
    // This routine is only used to get the full mask.
    int get_system_affinity(bool abort_on_error) override {
      KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),
                  "Illegal get affinity operation when not capable");

      // Set the mask with all CPUs that are available.
      for (int i = 0; i < __kmp_xproc; ++i)
        KMP_CPU_SET(i, this);
      return 0;
    }
    int set_system_affinity(bool abort_on_error) const override {
      KMP_ASSERT2(KMP_AFFINITY_CAPABLE(),

                  "Illegal set affinity operation when not capable");

      int location;
      int gtid = __kmp_entry_gtid();
      int tid = thread_self();

      // Unbind the thread if it was bound to any processors before so that
      // we can bind the thread to CPUs specified by the mask not others.
      int retval = bindprocessor(BINDTHREAD, tid, PROCESSOR_CLASS_ANY);

      // On AIX, we can only bind to one instead of a set of CPUs with the
      // bindprocessor() system call.
      KMP_CPU_SET_ITERATE(location, this) {
        if (KMP_CPU_ISSET(location, this)) {
          retval = bindprocessor(BINDTHREAD, tid, location);
          if (retval == -1 && errno == 1) {
            rsid_t rsid;
            rsethandle_t rsh;
            // Put something in rsh to prevent compiler warning
            // about uninitalized use
            rsh = rs_alloc(RS_EMPTY);
            rsid.at_pid = getpid();
            if (RS_DEFAULT_RSET != ra_getrset(R_PROCESS, rsid, 0, rsh)) {
              retval = ra_detachrset(R_PROCESS, rsid, 0);
              retval = bindprocessor(BINDTHREAD, tid, location);
            }
          }
          if (retval == 0) {
            KA_TRACE(10, ("__kmp_set_system_affinity:  Done binding "
                          "T#%d to cpu=%d.\n",
                          gtid, location));
            continue;
          }
          int error = errno;
          if (abort_on_error) {
            __kmp_fatal(KMP_MSG(FunctionError, "bindprocessor()"),
                        KMP_ERR(error), __kmp_msg_null);
            KA_TRACE(10, ("__kmp_set_system_affinity:  Error binding "
                          "T#%d to cpu=%d, errno=%d.\n",
                          gtid, location, error));
            return error;
          }
        }
      }
      return 0;
    }
#endif // !KMP_OS_AIX
  };
  void determine_capable(const char *env_var) override {
    __kmp_affinity_determine_capable(env_var);
  }
  void bind_thread(int which) override { __kmp_affinity_bind_thread(which); }
  KMPAffinity::Mask *allocate_mask() override {
    KMPNativeAffinity::Mask *retval = new Mask();
    return retval;
  }
  void deallocate_mask(KMPAffinity::Mask *m) override {
    KMPNativeAffinity::Mask *native_mask =
        static_cast<KMPNativeAffinity::Mask *>(m);
    delete native_mask;
  }
  KMPAffinity::Mask *allocate_mask_array(int num) override {
    return new Mask[num];
  }
  void deallocate_mask_array(KMPAffinity::Mask *array) override {
    Mask *linux_array = static_cast<Mask *>(array);
    delete[] linux_array;
  }
  KMPAffinity::Mask *index_mask_array(KMPAffinity::Mask *array,
                                      int index) override {
    Mask *linux_array = static_cast<Mask *>(array);
    return &(linux_array[index]);
  }
  api_type get_api_type() const override { return NATIVE_OS; }
};
#endif /* KMP_OS_LINUX || KMP_OS_FREEBSD || KMP_OS_NETBSD || KMP_OS_DRAGONFLY  \
          || KMP_OS_AIX */

#if KMP_OS_WINDOWS
class KMPNativeAffinity : public KMPAffinity {
  class Mask : public KMPAffinity::Mask {
    typedef ULONG_PTR mask_t;
    static const int BITS_PER_MASK_T = sizeof(mask_t) * CHAR_BIT;
    mask_t *mask;

  public:
    Mask() {
      mask = (mask_t *)__kmp_allocate(sizeof(mask_t) * __kmp_num_proc_groups);
    }
    ~Mask() {
      if (mask)
        __kmp_free(mask);
    }
    void set(int i) override {
      mask[i / BITS_PER_MASK_T] |= ((mask_t)1 << (i % BITS_PER_MASK_T));
    }
    bool is_set(int i) const override {
      return (mask[i / BITS_PER_MASK_T] & ((mask_t)1 << (i % BITS_PER_MASK_T)));
    }
    void clear(int i) override {
      mask[i / BITS_PER_MASK_T] &= ~((mask_t)1 << (i % BITS_PER_MASK_T));
    }
    void zero() override {
      for (int i = 0; i < __kmp_num_proc_groups; ++i)
        mask[i] = 0;
    }
    bool empty() const override {
      for (size_t i = 0; i < __kmp_num_proc_groups; ++i)
        if (mask[i])
          return false;
      return true;
    }
    void copy(const KMPAffinity::Mask *src) override {
      const Mask *convert = static_cast<const Mask *>(src);
      for (int i = 0; i < __kmp_num_proc_groups; ++i)
        mask[i] = convert->mask[i];
    }
    void bitwise_and(const KMPAffinity::Mask *rhs) override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      for (int i = 0; i < __kmp_num_proc_groups; ++i)
        mask[i] &= convert->mask[i];
    }
    void bitwise_or(const KMPAffinity::Mask *rhs) override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      for (int i = 0; i < __kmp_num_proc_groups; ++i)
        mask[i] |= convert->mask[i];
    }
    void bitwise_not() override {
      for (int i = 0; i < __kmp_num_proc_groups; ++i)
        mask[i] = ~(mask[i]);
    }
    bool is_equal(const KMPAffinity::Mask *rhs) const override {
      const Mask *convert = static_cast<const Mask *>(rhs);
      for (size_t i = 0; i < __kmp_num_proc_groups; ++i)
        if (mask[i] != convert->mask[i])
          return false;
      return true;
    }
    int begin() const override {
      int retval = 0;
      while (retval < end() && !is_set(retval))
        ++retval;
      return retval;
    }
    int end() const override { return __kmp_num_proc_groups * BITS_PER_MASK_T; }
    int next(int previous) const override {
      int retval = previous + 1;
      while (retval < end() && !is_set(retval))
        ++retval;
      return retval;
    }
    int set_process_affinity(bool abort_on_error) const override {
      if (__kmp_num_proc_groups <= 1) {
        if (!SetProcessAffinityMask(GetCurrentProcess(), *mask)) {
          DWORD error = GetLastError();
          if (abort_on_error) {
            __kmp_fatal(KMP_MSG(CantSetThreadAffMask), KMP_ERR(error),
                        __kmp_msg_null);
          }
          return error;
        }
      }
      return 0;
    }
    int set_system_affinity(bool abort_on_error) const override {
      if (__kmp_num_proc_groups > 1) {
        // Check for a valid mask.
        GROUP_AFFINITY ga;
        int group = get_proc_group();
        if (group < 0) {
          if (abort_on_error) {
            KMP_FATAL(AffinityInvalidMask, "kmp_set_affinity");
          }
          return -1;
        }
        // Transform the bit vector into a GROUP_AFFINITY struct
        // and make the system call to set affinity.
        ga.Group = group;
        ga.Mask = mask[group];
        ga.Reserved[0] = ga.Reserved[1] = ga.Reserved[2] = 0;

        KMP_DEBUG_ASSERT(__kmp_SetThreadGroupAffinity != NULL);
        if (__kmp_SetThreadGroupAffinity(GetCurrentThread(), &ga, NULL) == 0) {
          DWORD error = GetLastError();
          if (abort_on_error) {
            __kmp_fatal(KMP_MSG(CantSetThreadAffMask), KMP_ERR(error),
                        __kmp_msg_null);
          }
          return error;
        }
      } else {
        if (!SetThreadAffinityMask(GetCurrentThread(), *mask)) {
          DWORD error = GetLastError();
          if (abort_on_error) {
            __kmp_fatal(KMP_MSG(CantSetThreadAffMask), KMP_ERR(error),
                        __kmp_msg_null);
          }
          return error;
        }
      }
      return 0;
    }
    int get_system_affinity(bool abort_on_error) override {
      if (__kmp_num_proc_groups > 1) {
        this->zero();
        GROUP_AFFINITY ga;
        KMP_DEBUG_ASSERT(__kmp_GetThreadGroupAffinity != NULL);
        if (__kmp_GetThreadGroupAffinity(GetCurrentThread(), &ga) == 0) {
          DWORD error = GetLastError();
          if (abort_on_error) {
            __kmp_fatal(KMP_MSG(FunctionError, "GetThreadGroupAffinity()"),
                        KMP_ERR(error), __kmp_msg_null);
          }
          return error;
        }
        if ((ga.Group < 0) || (ga.Group > __kmp_num_proc_groups) ||
            (ga.Mask == 0)) {
          return -1;
        }
        mask[ga.Group] = ga.Mask;
      } else {
        mask_t newMask, sysMask, retval;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &newMask, &sysMask)) {
          DWORD error = GetLastError();
          if (abort_on_error) {
            __kmp_fatal(KMP_MSG(FunctionError, "GetProcessAffinityMask()"),
                        KMP_ERR(error), __kmp_msg_null);
          }
          return error;
        }
        retval = SetThreadAffinityMask(GetCurrentThread(), newMask);
        if (!retval) {
          DWORD error = GetLastError();
          if (abort_on_error) {
            __kmp_fatal(KMP_MSG(FunctionError, "SetThreadAffinityMask()"),
                        KMP_ERR(error), __kmp_msg_null);
          }
          return error;
        }
        newMask = SetThreadAffinityMask(GetCurrentThread(), retval);
        if (!newMask) {
          DWORD error = GetLastError();
          if (abort_on_error) {
            __kmp_fatal(KMP_MSG(FunctionError, "SetThreadAffinityMask()"),
                        KMP_ERR(error), __kmp_msg_null);
          }
        }
        *mask = retval;
      }
      return 0;
    }
    int get_proc_group() const override {
      int group = -1;
      if (__kmp_num_proc_groups == 1) {
        return 1;
      }
      for (int i = 0; i < __kmp_num_proc_groups; i++) {
        if (mask[i] == 0)
          continue;
        if (group >= 0)
          return -1;
        group = i;
      }
      return group;
    }
  };
  void determine_capable(const char *env_var) override {
    __kmp_affinity_determine_capable(env_var);
  }
  void bind_thread(int which) override { __kmp_affinity_bind_thread(which); }
  KMPAffinity::Mask *allocate_mask() override { return new Mask(); }
  void deallocate_mask(KMPAffinity::Mask *m) override { delete m; }
  KMPAffinity::Mask *allocate_mask_array(int num) override {
    return new Mask[num];
  }
  void deallocate_mask_array(KMPAffinity::Mask *array) override {
    Mask *windows_array = static_cast<Mask *>(array);
    delete[] windows_array;
  }
  KMPAffinity::Mask *index_mask_array(KMPAffinity::Mask *array,
                                      int index) override {
    Mask *windows_array = static_cast<Mask *>(array);
    return &(windows_array[index]);
  }
  api_type get_api_type() const override { return NATIVE_OS; }
};
#endif /* KMP_OS_WINDOWS */
#endif /* KMP_AFFINITY_SUPPORTED */

// Describe an attribute for a level in the machine topology
struct kmp_hw_attr_t {
  int core_type : 8;
  int core_eff : 8;
  unsigned valid : 1;
  unsigned reserved : 15;

  static const int UNKNOWN_CORE_EFF = -1;

  kmp_hw_attr_t()
      : core_type(KMP_HW_CORE_TYPE_UNKNOWN), core_eff(UNKNOWN_CORE_EFF),
        valid(0), reserved(0) {}
  void set_core_type(kmp_hw_core_type_t type) {
    valid = 1;
    core_type = type;
  }
  void set_core_eff(int eff) {
    valid = 1;
    core_eff = eff;
  }
  kmp_hw_core_type_t get_core_type() const {
    return (kmp_hw_core_type_t)core_type;
  }
  int get_core_eff() const { return core_eff; }
  bool is_core_type_valid() const {
    return core_type != KMP_HW_CORE_TYPE_UNKNOWN;
  }
  bool is_core_eff_valid() const { return core_eff != UNKNOWN_CORE_EFF; }
  operator bool() const { return valid; }
  void clear() {
    core_type = KMP_HW_CORE_TYPE_UNKNOWN;
    core_eff = UNKNOWN_CORE_EFF;
    valid = 0;
  }
  bool contains(const kmp_hw_attr_t &other) const {
    if (!valid && !other.valid)
      return true;
    if (valid && other.valid) {
      if (other.is_core_type_valid()) {
        if (!is_core_type_valid() || (get_core_type() != other.get_core_type()))
          return false;
      }
      if (other.is_core_eff_valid()) {
        if (!is_core_eff_valid() || (get_core_eff() != other.get_core_eff()))
          return false;
      }
      return true;
    }
    return false;
  }
#if KMP_AFFINITY_SUPPORTED
  bool contains(const kmp_affinity_attrs_t &attr) const {
    if (!valid && !attr.valid)
      return true;
    if (valid && attr.valid) {
      if (attr.core_type != KMP_HW_CORE_TYPE_UNKNOWN)
        return (is_core_type_valid() &&
                (get_core_type() == (kmp_hw_core_type_t)attr.core_type));
      if (attr.core_eff != UNKNOWN_CORE_EFF)
        return (is_core_eff_valid() && (get_core_eff() == attr.core_eff));
      return true;
    }
    return false;
  }
#endif // KMP_AFFINITY_SUPPORTED
  bool operator==(const kmp_hw_attr_t &rhs) const {
    return (rhs.valid == valid && rhs.core_eff == core_eff &&
            rhs.core_type == core_type);
  }
  bool operator!=(const kmp_hw_attr_t &rhs) const { return !operator==(rhs); }
};

#if KMP_AFFINITY_SUPPORTED
KMP_BUILD_ASSERT(sizeof(kmp_hw_attr_t) == sizeof(kmp_affinity_attrs_t));
#endif

class kmp_hw_thread_t {
public:
  static const int UNKNOWN_ID = -1;
  static const int MULTIPLE_ID = -2;
  static int compare_ids(const void *a, const void *b);
  static int compare_compact(const void *a, const void *b);
  int ids[KMP_HW_LAST];
  int sub_ids[KMP_HW_LAST];
  bool leader;
  int os_id;
  kmp_hw_attr_t attrs;

  void print() const;
  void clear() {
    for (int i = 0; i < (int)KMP_HW_LAST; ++i)
      ids[i] = UNKNOWN_ID;
    leader = false;
    attrs.clear();
  }
};

class kmp_topology_t {

  struct flags_t {
    int uniform : 1;
    int reserved : 31;
  };

  int depth;

  // The following arrays are all 'depth' long and have been
  // allocated to hold up to KMP_HW_LAST number of objects if
  // needed so layers can be added without reallocation of any array

  // Orderd array of the types in the topology
  kmp_hw_t *types;

  // Keep quick topology ratios, for non-uniform topologies,
  // this ratio holds the max number of itemAs per itemB
  // e.g., [ 4 packages | 6 cores / package | 2 threads / core ]
  int *ratio;

  // Storage containing the absolute number of each topology layer
  int *count;

  // The number of core efficiencies. This is only useful for hybrid
  // topologies. Core efficiencies will range from 0 to num efficiencies - 1
  int num_core_efficiencies;
  int num_core_types;
  kmp_hw_core_type_t core_types[KMP_HW_MAX_NUM_CORE_TYPES];

  // The hardware threads array
  // hw_threads is num_hw_threads long
  // Each hw_thread's ids and sub_ids are depth deep
  int num_hw_threads;
  kmp_hw_thread_t *hw_threads;

  // Equivalence hash where the key is the hardware topology item
  // and the value is the equivalent hardware topology type in the
  // types[] array, if the value is KMP_HW_UNKNOWN, then there is no
  // known equivalence for the topology type
  kmp_hw_t equivalent[KMP_HW_LAST];

  // Flags describing the topology
  flags_t flags;

  // Compact value used during sort_compact()
  int compact;

  // Insert a new topology layer after allocation
  void _insert_layer(kmp_hw_t type, const int *ids);

#if KMP_GROUP_AFFINITY
  // Insert topology information about Windows Processor groups
  void _insert_windows_proc_groups();
#endif

  // Count each item & get the num x's per y
  // e.g., get the number of cores and the number of threads per core
  // for each (x, y) in (KMP_HW_* , KMP_HW_*)
  void _gather_enumeration_information();

  // Remove layers that don't add information to the topology.
  // This is done by having the layer take on the id = UNKNOWN_ID (-1)
  void _remove_radix1_layers();

  // Find out if the topology is uniform
  void _discover_uniformity();

  // Set all the sub_ids for each hardware thread
  void _set_sub_ids();

  // Set global affinity variables describing the number of threads per
  // core, the number of packages, the number of cores per package, and
  // the number of cores.
  void _set_globals();

  // Set the last level cache equivalent type
  void _set_last_level_cache();

  // Return the number of cores with a particular attribute, 'attr'.
  // If 'find_all' is true, then find all cores on the machine, otherwise find
  // all cores per the layer 'above'
  int _get_ncores_with_attr(const kmp_hw_attr_t &attr, int above,
                            bool find_all = false) const;

public:
  // Force use of allocate()/deallocate()
  kmp_topology_t() = delete;
  kmp_topology_t(const kmp_topology_t &t) = delete;
  kmp_topology_t(kmp_topology_t &&t) = delete;
  kmp_topology_t &operator=(const kmp_topology_t &t) = delete;
  kmp_topology_t &operator=(kmp_topology_t &&t) = delete;

  static kmp_topology_t *allocate(int nproc, int ndepth, const kmp_hw_t *types);
  static void deallocate(kmp_topology_t *);

  // Functions used in create_map() routines
  kmp_hw_thread_t &at(int index) {
    KMP_DEBUG_ASSERT(index >= 0 && index < num_hw_threads);
    return hw_threads[index];
  }
  const kmp_hw_thread_t &at(int index) const {
    KMP_DEBUG_ASSERT(index >= 0 && index < num_hw_threads);
    return hw_threads[index];
  }
  int get_num_hw_threads() const { return num_hw_threads; }
  void sort_ids() {
    qsort(hw_threads, num_hw_threads, sizeof(kmp_hw_thread_t),
          kmp_hw_thread_t::compare_ids);
  }
  // Check if the hardware ids are unique, if they are
  // return true, otherwise return false
  bool check_ids() const;

  // Function to call after the create_map() routine
  void canonicalize();
  void canonicalize(int pkgs, int cores_per_pkg, int thr_per_core, int cores);

// Functions used after canonicalize() called

#if KMP_AFFINITY_SUPPORTED
  // Set the granularity for affinity settings
  void set_granularity(kmp_affinity_t &stgs) const;
  bool is_close(int hwt1, int hwt2, const kmp_affinity_t &stgs) const;
  bool restrict_to_mask(const kmp_affin_mask_t *mask);
  bool filter_hw_subset();
#endif
  bool is_uniform() const { return flags.uniform; }
  // Tell whether a type is a valid type in the topology
  // returns KMP_HW_UNKNOWN when there is no equivalent type
  kmp_hw_t get_equivalent_type(kmp_hw_t type) const {
    if (type == KMP_HW_UNKNOWN)
      return KMP_HW_UNKNOWN;
    return equivalent[type];
  }
  // Set type1 = type2
  void set_equivalent_type(kmp_hw_t type1, kmp_hw_t type2) {
    KMP_DEBUG_ASSERT_VALID_HW_TYPE(type1);
    KMP_DEBUG_ASSERT_VALID_HW_TYPE(type2);
    kmp_hw_t real_type2 = equivalent[type2];
    if (real_type2 == KMP_HW_UNKNOWN)
      real_type2 = type2;
    equivalent[type1] = real_type2;
    // This loop is required since any of the types may have been set to
    // be equivalent to type1.  They all must be checked and reset to type2.
    KMP_FOREACH_HW_TYPE(type) {
      if (equivalent[type] == type1) {
        equivalent[type] = real_type2;
      }
    }
  }
  // Calculate number of types corresponding to level1
  // per types corresponding to level2 (e.g., number of threads per core)
  int calculate_ratio(int level1, int level2) const {
    KMP_DEBUG_ASSERT(level1 >= 0 && level1 < depth);
    KMP_DEBUG_ASSERT(level2 >= 0 && level2 < depth);
    int r = 1;
    for (int level = level1; level > level2; --level)
      r *= ratio[level];
    return r;
  }
  int get_ratio(int level) const {
    KMP_DEBUG_ASSERT(level >= 0 && level < depth);
    return ratio[level];
  }
  int get_depth() const { return depth; };
  kmp_hw_t get_type(int level) const {
    KMP_DEBUG_ASSERT(level >= 0 && level < depth);
    return types[level];
  }
  int get_level(kmp_hw_t type) const {
    KMP_DEBUG_ASSERT_VALID_HW_TYPE(type);
    int eq_type = equivalent[type];
    if (eq_type == KMP_HW_UNKNOWN)
      return -1;
    for (int i = 0; i < depth; ++i)
      if (types[i] == eq_type)
        return i;
    return -1;
  }
  int get_count(int level) const {
    KMP_DEBUG_ASSERT(level >= 0 && level < depth);
    return count[level];
  }
  // Return the total number of cores with attribute 'attr'
  int get_ncores_with_attr(const kmp_hw_attr_t &attr) const {
    return _get_ncores_with_attr(attr, -1, true);
  }
  // Return the number of cores with attribute
  // 'attr' per topology level 'above'
  int get_ncores_with_attr_per(const kmp_hw_attr_t &attr, int above) const {
    return _get_ncores_with_attr(attr, above, false);
  }

#if KMP_AFFINITY_SUPPORTED
  friend int kmp_hw_thread_t::compare_compact(const void *a, const void *b);
  void sort_compact(kmp_affinity_t &affinity) {
    compact = affinity.compact;
    qsort(hw_threads, num_hw_threads, sizeof(kmp_hw_thread_t),
          kmp_hw_thread_t::compare_compact);
  }
#endif
  void print(const char *env_var = "KMP_AFFINITY") const;
  void dump() const;
};
extern kmp_topology_t *__kmp_topology;

class kmp_hw_subset_t {
  const static size_t MAX_ATTRS = KMP_HW_MAX_NUM_CORE_EFFS;

public:
  // Describe a machine topology item in KMP_HW_SUBSET
  struct item_t {
    kmp_hw_t type;
    int num_attrs;
    int num[MAX_ATTRS];
    int offset[MAX_ATTRS];
    kmp_hw_attr_t attr[MAX_ATTRS];
  };
  // Put parenthesis around max to avoid accidental use of Windows max macro.
  const static int USE_ALL = (std::numeric_limits<int>::max)();

private:
  int depth;
  int capacity;
  item_t *items;
  kmp_uint64 set;
  bool absolute;
  // The set must be able to handle up to KMP_HW_LAST number of layers
  KMP_BUILD_ASSERT(sizeof(set) * 8 >= KMP_HW_LAST);
  // Sorting the KMP_HW_SUBSET items to follow topology order
  // All unknown topology types will be at the beginning of the subset
  static int hw_subset_compare(const void *i1, const void *i2) {
    kmp_hw_t type1 = ((const item_t *)i1)->type;
    kmp_hw_t type2 = ((const item_t *)i2)->type;
    int level1 = __kmp_topology->get_level(type1);
    int level2 = __kmp_topology->get_level(type2);
    return level1 - level2;
  }

public:
  // Force use of allocate()/deallocate()
  kmp_hw_subset_t() = delete;
  kmp_hw_subset_t(const kmp_hw_subset_t &t) = delete;
  kmp_hw_subset_t(kmp_hw_subset_t &&t) = delete;
  kmp_hw_subset_t &operator=(const kmp_hw_subset_t &t) = delete;
  kmp_hw_subset_t &operator=(kmp_hw_subset_t &&t) = delete;

  static kmp_hw_subset_t *allocate() {
    int initial_capacity = 5;
    kmp_hw_subset_t *retval =
        (kmp_hw_subset_t *)__kmp_allocate(sizeof(kmp_hw_subset_t));
    retval->depth = 0;
    retval->capacity = initial_capacity;
    retval->set = 0ull;
    retval->absolute = false;
    retval->items = (item_t *)__kmp_allocate(sizeof(item_t) * initial_capacity);
    return retval;
  }
  static void deallocate(kmp_hw_subset_t *subset) {
    __kmp_free(subset->items);
    __kmp_free(subset);
  }
  void set_absolute() { absolute = true; }
  bool is_absolute() const { return absolute; }
  void push_back(int num, kmp_hw_t type, int offset, kmp_hw_attr_t attr) {
    for (int i = 0; i < depth; ++i) {
      // Found an existing item for this layer type
      // Add the num, offset, and attr to this item
      if (items[i].type == type) {
        int idx = items[i].num_attrs++;
        if ((size_t)idx >= MAX_ATTRS)
          return;
        items[i].num[idx] = num;
        items[i].offset[idx] = offset;
        items[i].attr[idx] = attr;
        return;
      }
    }
    if (depth == capacity - 1) {
      capacity *= 2;
      item_t *new_items = (item_t *)__kmp_allocate(sizeof(item_t) * capacity);
      for (int i = 0; i < depth; ++i)
        new_items[i] = items[i];
      __kmp_free(items);
      items = new_items;
    }
    items[depth].num_attrs = 1;
    items[depth].type = type;
    items[depth].num[0] = num;
    items[depth].offset[0] = offset;
    items[depth].attr[0] = attr;
    depth++;
    set |= (1ull << type);
  }
  int get_depth() const { return depth; }
  const item_t &at(int index) const {
    KMP_DEBUG_ASSERT(index >= 0 && index < depth);
    return items[index];
  }
  item_t &at(int index) {
    KMP_DEBUG_ASSERT(index >= 0 && index < depth);
    return items[index];
  }
  void remove(int index) {
    KMP_DEBUG_ASSERT(index >= 0 && index < depth);
    set &= ~(1ull << items[index].type);
    for (int j = index + 1; j < depth; ++j) {
      items[j - 1] = items[j];
    }
    depth--;
  }
  void sort() {
    KMP_DEBUG_ASSERT(__kmp_topology);
    qsort(items, depth, sizeof(item_t), hw_subset_compare);
  }
  bool specified(kmp_hw_t type) const { return ((set & (1ull << type)) > 0); }
  void dump() const {
    printf("**********************\n");
    printf("*** kmp_hw_subset: ***\n");
    printf("* depth: %d\n", depth);
    printf("* items:\n");
    for (int i = 0; i < depth; ++i) {
      printf(" type: %s\n", __kmp_hw_get_keyword(items[i].type));
      for (int j = 0; j < items[i].num_attrs; ++j) {
        printf("  num: %d, offset: %d, attr: ", items[i].num[j],
               items[i].offset[j]);
        if (!items[i].attr[j]) {
          printf(" (none)\n");
        } else {
          printf(
              " core_type = %s, core_eff = %d\n",
              __kmp_hw_get_core_type_string(items[i].attr[j].get_core_type()),
              items[i].attr[j].get_core_eff());
        }
      }
    }
    printf("* set: 0x%llx\n", set);
    printf("* absolute: %d\n", absolute);
    printf("**********************\n");
  }
};
extern kmp_hw_subset_t *__kmp_hw_subset;

/* A structure for holding machine-specific hierarchy info to be computed once
   at init. This structure represents a mapping of threads to the actual machine
   hierarchy, or to our best guess at what the hierarchy might be, for the
   purpose of performing an efficient barrier. In the worst case, when there is
   no machine hierarchy information, it produces a tree suitable for a barrier,
   similar to the tree used in the hyper barrier. */
class hierarchy_info {
public:
  /* Good default values for number of leaves and branching factor, given no
     affinity information. Behaves a bit like hyper barrier. */
  static const kmp_uint32 maxLeaves = 4;
  static const kmp_uint32 minBranch = 4;
  /** Number of levels in the hierarchy. Typical levels are threads/core,
      cores/package or socket, packages/node, nodes/machine, etc. We don't want
      to get specific with nomenclature. When the machine is oversubscribed we
      add levels to duplicate the hierarchy, doubling the thread capacity of the
      hierarchy each time we add a level. */
  kmp_uint32 maxLevels;

  /** This is specifically the depth of the machine configuration hierarchy, in
      terms of the number of levels along the longest path from root to any
      leaf. It corresponds to the number of entries in numPerLevel if we exclude
      all but one trailing 1. */
  kmp_uint32 depth;
  kmp_uint32 base_num_threads;
  enum init_status { initialized = 0, not_initialized = 1, initializing = 2 };
  volatile kmp_int8 uninitialized; // 0=initialized, 1=not initialized,
  // 2=initialization in progress
  volatile kmp_int8 resizing; // 0=not resizing, 1=resizing

  /** Level 0 corresponds to leaves. numPerLevel[i] is the number of children
      the parent of a node at level i has. For example, if we have a machine
      with 4 packages, 4 cores/package and 2 HT per core, then numPerLevel =
      {2, 4, 4, 1, 1}. All empty levels are set to 1. */
  kmp_uint32 *numPerLevel;
  kmp_uint32 *skipPerLevel;

  void deriveLevels() {
    int hier_depth = __kmp_topology->get_depth();
    for (int i = hier_depth - 1, level = 0; i >= 0; --i, ++level) {
      numPerLevel[level] = __kmp_topology->get_ratio(i);
    }
  }

  hierarchy_info()
      : maxLevels(7), depth(1), uninitialized(not_initialized), resizing(0) {}

  void fini() {
    if (!uninitialized && numPerLevel) {
      __kmp_free(numPerLevel);
      numPerLevel = NULL;
      uninitialized = not_initialized;
    }
  }

  void init(int num_addrs) {
    kmp_int8 bool_result = KMP_COMPARE_AND_STORE_ACQ8(
        &uninitialized, not_initialized, initializing);
    if (bool_result == 0) { // Wait for initialization
      while (TCR_1(uninitialized) != initialized)
        KMP_CPU_PAUSE();
      return;
    }
    KMP_DEBUG_ASSERT(bool_result == 1);

    /* Added explicit initialization of the data fields here to prevent usage of
       dirty value observed when static library is re-initialized multiple times
       (e.g. when non-OpenMP thread repeatedly launches/joins thread that uses
       OpenMP). */
    depth = 1;
    resizing = 0;
    maxLevels = 7;
    numPerLevel =
        (kmp_uint32 *)__kmp_allocate(maxLevels * 2 * sizeof(kmp_uint32));
    skipPerLevel = &(numPerLevel[maxLevels]);
    for (kmp_uint32 i = 0; i < maxLevels;
         ++i) { // init numPerLevel[*] to 1 item per level
      numPerLevel[i] = 1;
      skipPerLevel[i] = 1;
    }

    // Sort table by physical ID
    if (__kmp_topology && __kmp_topology->get_depth() > 0) {
      deriveLevels();
    } else {
      numPerLevel[0] = maxLeaves;
      numPerLevel[1] = num_addrs / maxLeaves;
      if (num_addrs % maxLeaves)
        numPerLevel[1]++;
    }

    base_num_threads = num_addrs;
    for (int i = maxLevels - 1; i >= 0;
         --i) // count non-empty levels to get depth
      if (numPerLevel[i] != 1 || depth > 1) // only count one top-level '1'
        depth++;

    kmp_uint32 branch = minBranch;
    if (numPerLevel[0] == 1)
      branch = num_addrs / maxLeaves;
    if (branch < minBranch)
      branch = minBranch;
    for (kmp_uint32 d = 0; d < depth - 1; ++d) { // optimize hierarchy width
      while (numPerLevel[d] > branch ||
             (d == 0 && numPerLevel[d] > maxLeaves)) { // max 4 on level 0!
        if (numPerLevel[d] & 1)
          numPerLevel[d]++;
        numPerLevel[d] = numPerLevel[d] >> 1;
        if (numPerLevel[d + 1] == 1)
          depth++;
        numPerLevel[d + 1] = numPerLevel[d + 1] << 1;
      }
      if (numPerLevel[0] == 1) {
        branch = branch >> 1;
        if (branch < 4)
          branch = minBranch;
      }
    }

    for (kmp_uint32 i = 1; i < depth; ++i)
      skipPerLevel[i] = numPerLevel[i - 1] * skipPerLevel[i - 1];
    // Fill in hierarchy in the case of oversubscription
    for (kmp_uint32 i = depth; i < maxLevels; ++i)
      skipPerLevel[i] = 2 * skipPerLevel[i - 1];

    uninitialized = initialized; // One writer
  }

  // Resize the hierarchy if nproc changes to something larger than before
  void resize(kmp_uint32 nproc) {
    kmp_int8 bool_result = KMP_COMPARE_AND_STORE_ACQ8(&resizing, 0, 1);
    while (bool_result == 0) { // someone else is trying to resize
      KMP_CPU_PAUSE();
      if (nproc <= base_num_threads) // happy with other thread's resize
        return;
      else // try to resize
        bool_result = KMP_COMPARE_AND_STORE_ACQ8(&resizing, 0, 1);
    }
    KMP_DEBUG_ASSERT(bool_result != 0);
    if (nproc <= base_num_threads)
      return; // happy with other thread's resize

    // Calculate new maxLevels
    kmp_uint32 old_sz = skipPerLevel[depth - 1];
    kmp_uint32 incs = 0, old_maxLevels = maxLevels;
    // First see if old maxLevels is enough to contain new size
    for (kmp_uint32 i = depth; i < maxLevels && nproc > old_sz; ++i) {
      skipPerLevel[i] = 2 * skipPerLevel[i - 1];
      numPerLevel[i - 1] *= 2;
      old_sz *= 2;
      depth++;
    }
    if (nproc > old_sz) { // Not enough space, need to expand hierarchy
      while (nproc > old_sz) {
        old_sz *= 2;
        incs++;
        depth++;
      }
      maxLevels += incs;

      // Resize arrays
      kmp_uint32 *old_numPerLevel = numPerLevel;
      kmp_uint32 *old_skipPerLevel = skipPerLevel;
      numPerLevel = skipPerLevel = NULL;
      numPerLevel =
          (kmp_uint32 *)__kmp_allocate(maxLevels * 2 * sizeof(kmp_uint32));
      skipPerLevel = &(numPerLevel[maxLevels]);

      // Copy old elements from old arrays
      for (kmp_uint32 i = 0; i < old_maxLevels; ++i) {
        // init numPerLevel[*] to 1 item per level
        numPerLevel[i] = old_numPerLevel[i];
        skipPerLevel[i] = old_skipPerLevel[i];
      }

      // Init new elements in arrays to 1
      for (kmp_uint32 i = old_maxLevels; i < maxLevels; ++i) {
        // init numPerLevel[*] to 1 item per level
        numPerLevel[i] = 1;
        skipPerLevel[i] = 1;
      }

      // Free old arrays
      __kmp_free(old_numPerLevel);
    }

    // Fill in oversubscription levels of hierarchy
    for (kmp_uint32 i = old_maxLevels; i < maxLevels; ++i)
      skipPerLevel[i] = 2 * skipPerLevel[i - 1];

    base_num_threads = nproc;
    resizing = 0; // One writer
  }
};
#endif // KMP_AFFINITY_H
