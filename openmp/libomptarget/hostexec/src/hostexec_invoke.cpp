
#include <stdint.h>

#define GLOB_ATTR __attribute__((address_space(1)))
#define __static_inl static __attribute__((flatten, always_inline))
#define __inl __attribute__((flatten, always_inline))

// mem order codes: A=acquire, X=relaxed, R=release

// headers for amdgcn opencl atomics
extern "C" __inl uint64_t oclAtomic64Load_A(GLOB_ATTR uint64_t *Address);
extern "C" __inl uint64_t oclAtomic64Load_X(GLOB_ATTR uint64_t *Address);
extern "C" __inl uint32_t oclAtomic32Load_A(GLOB_ATTR const uint32_t *Address);
extern "C" __inl uint32_t oclAtomic64CAS_AX(GLOB_ATTR uint64_t *Address,
                                            uint64_t *e_Val, uint64_t new_ptr);
extern "C" __inl uint32_t oclAtomic64CAS_RX(GLOB_ATTR uint64_t *Address,
                                            uint64_t *e_Val, uint64_t new_ptr);

// headers for cuda nvptx atomics
extern "C" __attribute__((nothrow)) unsigned long long
__ullAtomicAdd_system(unsigned long long *address, unsigned long long val);
extern "C" __attribute__((nothrow)) unsigned long long
__ullAtomicCAS_system(unsigned long long int *address,
                      unsigned long long int compare,
                      unsigned long long int val);
// headers for builtins
int __builtin_popcountl(unsigned long);
int __builtin_popcount(unsigned);
int __builtin_amdgcn_readfirstlane(int);
unsigned long __builtin_amdgcn_read_exec();
unsigned int __builtin_amdgcn_mbcnt_hi(unsigned int, unsigned int);
unsigned int __builtin_amdgcn_mbcnt_lo(unsigned int, unsigned int);
int __nvvm_read_ptx_sreg_tid_x();

// We need __ockl_hsa_signal_add and __ockl_lane_u32 from ockl
typedef uint64_t hsa_signal_value_t;
typedef uint64_t hsa_signal_t;
// typedef struct hsa_signal_s {
// uint64_t handle;
//} hsa_signal_t;
typedef enum __ockl_memory_order_e {
  __ockl_memory_order_relaxed = __ATOMIC_RELAXED,
  __ockl_memory_order_acquire = __ATOMIC_ACQUIRE,
  __ockl_memory_order_release = __ATOMIC_RELEASE,
  __ockl_memory_order_acq_rel = __ATOMIC_ACQ_REL,
  __ockl_memory_order_seq_cst = __ATOMIC_SEQ_CST,
} __ockl_memory_order;
extern "C" void __ockl_hsa_signal_add(hsa_signal_t signal,
                                      hsa_signal_value_t value,
                                      __ockl_memory_order mo);
extern "C" uint32_t __ockl_lane_u32();

#pragma omp begin declare target device_type(nohost)

typedef enum { STATUS_SUCCESS, STATUS_BUSY } status_t;

typedef enum {
  CONTROL_OFFSET_READY_FLAG = 0,
  CONTROL_OFFSET_RESERVED0 = 1,
} control_offset_t;

typedef enum {
  CONTROL_WIDTH_READY_FLAG = 1,
  CONTROL_WIDTH_RESERVED0 = 31,
} control_width_t;

typedef uint64_t LaneMask_t;

typedef struct {
  uint64_t next;
  LaneMask_t activemask;
  uint32_t service;
  uint32_t control;
} header_t;

typedef struct {
  // 64 slots of 8 uint64_ts each (4KB/payload)
  uint64_t slots[64][8];
} payload_t;

typedef struct {
  GLOB_ATTR header_t *headers;
  GLOB_ATTR payload_t *payloads;
  hsa_signal_t doorbell;
  uint64_t free_stack;
  uint64_t ready_stack;
  uint32_t index_size;
  uint32_t device_id;
} buffer_t;

namespace impl {

// These functions have arch-specific variants
__inl void deviceSleepHostWait();
//  These still dont have nvptx variants
__inl void send_signal(hsa_signal_t signal);
__inl uint32_t first_lane_id(uint32_t val);
__inl uint32_t lane_id();
__inl uint64_t get_mask();
__inl uint64_t atomic64Load_A(GLOB_ATTR uint64_t *Address);
__inl uint64_t atomic64Load_X(GLOB_ATTR uint64_t *Address);
__inl uint32_t atomic32Load_A(GLOB_ATTR const uint32_t *Address);
__inl bool atomic64CAS_AX(GLOB_ATTR uint64_t *Address, uint64_t *e_Val,
                          uint64_t new_ptr);
__inl bool atomic64CAS_RX(GLOB_ATTR uint64_t *Address, uint64_t *e_Val,
                          uint64_t new_ptr);
__inl void write_needs_host_services_symbol();

#pragma omp begin declare variant match(device = {arch(amdgcn)})

__static_inl void write_needs_host_services_symbol() {
  // The global variable "__needs_host_services" is used to detect that
  // host services are required. If hostexec_invoke is not called, the symbol
  // will not be present and the runtime can avoid allocating and initialising
  // service_thread_buf.
  __asm__(".type __needs_host_services,@object\n\t"
          ".global __needs_host_services\n\t"
          ".comm __needs_host_services,4" ::
              :);
}

__static_inl uint32_t lane_id() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
};

__static_inl uint32_t first_lane_id(uint32_t me) {
  return __builtin_amdgcn_readfirstlane(me);
}
__static_inl uint64_t get_mask() { return __builtin_amdgcn_read_exec(); }
__static_inl void send_signal(hsa_signal_t signal) {
  __ockl_hsa_signal_add(signal, 1, __ockl_memory_order_release);
}
__static_inl void deviceSleepHostWait() { __builtin_amdgcn_s_sleep(1); }
__static_inl uint64_t atomic64Load_A(GLOB_ATTR uint64_t *Address) {
  return oclAtomic64Load_A(Address);
}
__static_inl uint64_t atomic64Load_X(GLOB_ATTR uint64_t *Address) {
  return oclAtomic64Load_X(Address);
}
__static_inl uint32_t atomic32Load_A(GLOB_ATTR const uint32_t *Address) {
  return oclAtomic32Load_A(Address);
}
__static_inl bool atomic64CAS_AX(GLOB_ATTR uint64_t *Address, uint64_t *e_Val,
                                 uint64_t new_ptr) {
  return (bool)oclAtomic64CAS_AX(Address, e_Val, new_ptr);
}
__static_inl bool atomic64CAS_RX(GLOB_ATTR uint64_t *Address, uint64_t *e_Val,
                                 uint64_t new_ptr) {
  return (bool)oclAtomic64CAS_RX(Address, e_Val, new_ptr);
}

#pragma omp end declare variant

#pragma omp begin declare variant match(                                       \
        device = {arch(nvptx, nvptx64)},                                       \
            implementation = {extension(match_any)})

__static_inl void write_needs_host_services_symbol() {
  // The global variable "__needs_host_services" is used to detect that
  // host services are required. If hostexec_invoke is not called, the symbol
  // will not be present and the runtime can avoid allocating and initialising
  // service_thread_buf.
  __asm__(".global .align 4 .u32 __needs_host_services = 1;");
}
__static_inl inline void send_signal(hsa_signal_t signal) {
  __ullAtomicAdd_system((unsigned long long *)signal, 1);
}

__static_inl void deviceSleepHostWait() {
  int32_t start = __nvvm_read_ptx_sreg_clock();
  for (;;) {
    if ((__nvvm_read_ptx_sreg_clock() - start) >= 1000)
      break;
  }
}

__static_inl uint64_t get_mask() {
  unsigned int Mask;
  asm("activemask.b32 %0;" : "=r"(Mask));
  uint64_t mask64 = (uint64_t)Mask;
  return mask64;
}

//  FIXME: nvptx needs to use lane_id somehow here
__static_inl uint32_t first_lane_id(unsigned int lane_id) {
  unsigned int mask = (unsigned int)get_mask();
  if (mask == 0)
    return 0;
  unsigned int pos = 0;
  unsigned int m = 1;
  while (!(mask & m)) {
    m = m << 1;
    pos++;
  }
  return pos;
};

__static_inl uint32_t lane_id() {
  return (uint32_t)(__nvvm_read_ptx_sreg_tid_x() & 31);
};

__static_inl uint64_t atomic64Load_A(GLOB_ATTR uint64_t *Address) {
  unsigned long long result =
      __ullAtomicAdd_system((unsigned long long *)Address, 0);
  return (uint64_t)result;
}
__static_inl uint64_t atomic64Load_X(GLOB_ATTR uint64_t *Address) {
  unsigned long long result =
      __ullAtomicAdd_system((unsigned long long *)Address, 0);
  return (uint64_t)result;
}
__static_inl uint32_t atomic32Load_A(GLOB_ATTR const uint32_t *Address) {
  return __uAtomicAdd((uint32_t *)Address, 0);
}
__static_inl bool atomic64CAS_AX(GLOB_ATTR uint64_t *Address, uint64_t *e_Val,
                                 uint64_t new_ptr) {
  unsigned long long result = __ullAtomicCAS_system(
      (unsigned long long *)Address, (unsigned long long)*e_Val,
      (unsigned long long)new_ptr);
  return (bool)result;
}
__static_inl bool atomic64CAS_RX(GLOB_ATTR uint64_t *Address, uint64_t *e_Val,
                                 uint64_t new_ptr) {
  unsigned long long result = __ullAtomicCAS_system(
      (unsigned long long *)Address, (unsigned long long)*e_Val,
      (unsigned long long)new_ptr);
  return (bool)result;
}

#pragma omp end declare variant

} // end namespace impl

__static_inl uint64_t get_ptr_index(uint64_t ptr, uint32_t index_size) {
  return ptr & (((uint64_t)1 << index_size) - 1);
}

__static_inl GLOB_ATTR header_t *get_header(GLOB_ATTR buffer_t *buffer,
                                            uint64_t ptr) {
  return buffer->headers + get_ptr_index(ptr, buffer->index_size);
}

__static_inl GLOB_ATTR payload_t *get_payload(GLOB_ATTR buffer_t *buffer,
                                              uint64_t ptr) {
  return buffer->payloads + get_ptr_index(ptr, buffer->index_size);
}

// get_control_field only used by get_ready_flag
__static_inl uint32_t get_control_field(uint32_t control, uint32_t offset,
                                        uint32_t width) {
  return (control >> offset) & ((1 << width) - 1);
}

// get_ready_flag only called by lead lane of get_return_value
//                on atomically loaded control field of packet header
__static_inl uint32_t get_ready_flag(uint32_t control) {
  return get_control_field(control, CONTROL_OFFSET_READY_FLAG,
                           CONTROL_WIDTH_READY_FLAG);
}

// set_control_field only used by set_ready_flag
__static_inl uint32_t set_control_field(uint32_t control, uint32_t offset,
                                        uint32_t width, uint32_t value) {
  uint32_t mask = ~(((1 << width) - 1) << offset);
  return (control & mask) | (value << offset);
}

__static_inl uint32_t set_ready_flag(uint32_t control) {
  return set_control_field(control, CONTROL_OFFSET_READY_FLAG,
                           CONTROL_WIDTH_READY_FLAG, 1);
}

__static_inl uint64_t pop(GLOB_ATTR uint64_t *top, GLOB_ATTR buffer_t *buffer) {
  uint64_t F = impl::atomic64Load_A(top);
  // F is guaranteed to be non-zero, since there are at least as
  // many packets as there are waves, and each wave can hold at most
  // one packet.
  while (true) {
    GLOB_ATTR header_t *P = get_header(buffer, F);
    uint64_t N = impl::atomic64Load_X(&P->next);
    if (impl::atomic64CAS_AX(top, &F, N))
      break;
    impl::deviceSleepHostWait();
  }

  return F;
}

/** \brief Use the first active lane to get a free packet and
 *         broadcast to the whole wave.
 */
__static_inl uint64_t pop_free_stack(GLOB_ATTR buffer_t *buffer, uint32_t me,
                                     uint32_t low) {
  uint64_t packet_ptr = 0;
  if (me == low) {
    packet_ptr = pop(&buffer->free_stack, buffer);
  }

  uint32_t ptr_lo = packet_ptr;
  uint32_t ptr_hi = packet_ptr >> 32;
  ptr_lo = impl::first_lane_id(ptr_lo);
  ptr_hi = impl::first_lane_id(ptr_hi);

  return ((uint64_t)ptr_hi << 32) | ptr_lo;
}

__static_inl void push(GLOB_ATTR uint64_t *top, uint64_t ptr,
                       GLOB_ATTR buffer_t *buffer) {
  uint64_t F = impl::atomic64Load_X(top);
  GLOB_ATTR header_t *P = get_header(buffer, ptr);

  while (true) {
    P->next = F;
    if (impl::atomic64CAS_RX(top, &F, ptr))
      break;
    impl::deviceSleepHostWait();
  }
}

/** \brief Use the first active lane in a wave to submit a ready
 *         packet and signal the host.
 */
__static_inl void push_ready_stack(GLOB_ATTR buffer_t *buffer, uint64_t ptr,
                                   uint32_t me, uint32_t low) {
  if (me == low) {
    push(&buffer->ready_stack, ptr, buffer);
    impl::send_signal(buffer->doorbell);
  }
}

__static_inl uint64_t inc_ptr_tag(uint64_t ptr, uint32_t index_size) {
  // Unit step for the tag.
  uint64_t inc = 1UL << index_size;
  ptr += inc;
  // When the tag for index 0 wraps, increment the tag.
  return ptr == 0 ? inc : ptr;
}

/** \brief Return the packet after incrementing the ABA tag
 */
__static_inl void return_free_packet(GLOB_ATTR buffer_t *buffer, uint64_t ptr,
                                     uint32_t me, uint32_t low) {
  if (me == low) {
    ptr = inc_ptr_tag(ptr, buffer->index_size);
    push(&buffer->free_stack, ptr, buffer);
  }
}

void __static_inl fill_packet(GLOB_ATTR header_t *header,
                              GLOB_ATTR payload_t *payload, uint32_t service_id,
                              uint64_t arg0, uint64_t arg1, uint64_t arg2,
                              uint64_t arg3, uint64_t arg4, uint64_t arg5,
                              uint64_t arg6, uint64_t arg7, uint32_t me,
                              uint32_t low) {
  uint64_t active = impl::get_mask();
  if (me == low) {
    header->service = service_id;
    header->activemask = active;
    uint32_t control = set_ready_flag(0);
    header->control = control;
  }
  GLOB_ATTR uint64_t *ptr = payload->slots[me];
  ptr[0] = arg0;
  ptr[1] = arg1;
  ptr[2] = arg2;
  ptr[3] = arg3;
  ptr[4] = arg4;
  ptr[5] = arg5;
  ptr[6] = arg6;
  ptr[7] = arg7;
}

//  result is 8*8=64 bytes per lane
//  Total payload could be 64 lanes * 64 bytes = 4KB
typedef struct {
  uint64_t arg0;
  uint64_t arg1;
  uint64_t arg2;
  uint64_t arg3;
  uint64_t arg4;
  uint64_t arg5;
  uint64_t arg6;
  uint64_t arg7;
} hostexec_result_t;

/** \brief Wait for the host response and return the first two uint64_t
 *         entries per workitem.
 *
 *  After the packet is submitted in READY state, the wave spins until
 *  the host changes the state to DONE. Each workitem reads the first
 *  two uint64_t elements in its slot and returns this.
 */
__static_inl hostexec_result_t get_return_value(GLOB_ATTR header_t *header,
                                                GLOB_ATTR payload_t *payload,
                                                uint32_t me, uint32_t low) {
  // The while loop needs to be executed by all active
  // lanes. Otherwise, later reads from ptr are performed only by
  // the first thread, while other threads reuse a value cached from
  // previous operations. The use of readfirstlane in the while loop
  // prevents this reordering.
  //
  // In the absence of the readfirstlane, only one thread has a
  // sequenced-before relation from the atomic load on
  // header->control to the ordinary loads on ptr. As a result, the
  // compiler is free to reorder operations in such a way that the
  // ordinary loads are performed only by the first thread. The use
  // of readfirstlane provides a stronger code-motion barrier, and
  // it effectively "spreads out" the sequenced-before relation to
  // the ordinary stores in other threads too.
  while (true) {
    uint32_t ready_flag = 1;
    if (me == low) {
      uint32_t control =
          impl::atomic32Load_A((GLOB_ATTR uint32_t *)&header->control);
      ready_flag = get_ready_flag(control);
    }
    ready_flag = impl::first_lane_id(ready_flag);
    if (ready_flag == 0)
      break;
    impl::deviceSleepHostWait();
  }

  GLOB_ATTR uint64_t *ptr = (GLOB_ATTR uint64_t *)(payload->slots + me);
  hostexec_result_t retval;
  retval.arg0 = *ptr++;
  retval.arg1 = *ptr++;
  retval.arg2 = *ptr++;
  retval.arg3 = *ptr++;
  retval.arg4 = *ptr++;
  retval.arg5 = *ptr++;
  retval.arg6 = *ptr++;
  retval.arg7 = *ptr;

  return retval;
}

#undef __static_inl

/** \brief The implementation that should be hidden behind an ABI
 *
 *  The transaction is a wave-wide operation, where the service_id
 *  must be uniform, but the parameters are different for each
 *  workitem. Parameters from all active lanes are written into a
 *  hostcall packet. The hostcall blocks until the host processes the
 *  request, and returns the response it receiveds.
 *
 *  TODO: This function and everything above it should eventually move
 *  to a separate library that is loaded by the language runtime. The
 *  function itself will be exposed as an orindary function symbol to
 *  be linked into kernel objects that are loaded after this library.
 */

//  service_thread_buf is a global constant symbol that contains the
//  pointer to the buffer used by the service thread. This is written
//  by nextgen plugin method writeGlobalToDevice only when the device
//  image requires host services.
//  This is the alternative to using reserved IMPLICIT kern arg hostcall
//  because nvptx arch does not have implicit kern args.
uint64_t [[clang::address_space(4)]] service_thread_buf
    [[clang::loader_uninitialized]] __attribute__((used, retain, weak,
                                                   visibility("protected")));

extern "C" __attribute__((noinline)) hostexec_result_t
hostexec_invoke(const uint32_t service_id, uint64_t arg0, uint64_t arg1,
                uint64_t arg2, uint64_t arg3, uint64_t arg4, uint64_t arg5,
                uint64_t arg6, uint64_t arg7) {
  impl::write_needs_host_services_symbol();
  uint32_t me = impl::lane_id();
  uint32_t low = impl::first_lane_id(me);

  GLOB_ATTR buffer_t *buffer = (GLOB_ATTR buffer_t *)service_thread_buf;

  uint64_t packet_ptr = pop_free_stack(buffer, me, low);
  GLOB_ATTR header_t *header = get_header(buffer, packet_ptr);
  GLOB_ATTR payload_t *payload = get_payload(buffer, packet_ptr);
  fill_packet(header, payload, service_id, arg0, arg1, arg2, arg3, arg4, arg5,
              arg6, arg7, me, low);
  push_ready_stack(buffer, packet_ptr, me, low);
  hostexec_result_t retval = get_return_value(header, payload, me, low);
  return_free_packet(buffer, packet_ptr, me, low);
  return retval;
}

#pragma omp end declare target
