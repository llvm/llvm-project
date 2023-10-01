//===---- amdgcn_hostrpc.cpp - Services thread management  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code for the services thread for the hostrpc system using
// amdgcn hsa. This nvptx cuda variant of this process is in development.
//
//===----------------------------------------------------------------------===//

#include "../src/hostexec_internal.h"
#include "execute_service.h"
#include "urilocator.h"
#include <cassert>
#include <atomic>
#include <cstring>
#include <functional>
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>

/// Defines how many GPUs are maximally supported on a system
#define AMD_MAX_HSA_AGENTS 16

/** Opaque wrapper for signal */
typedef struct {
  uint64_t handle;
} signal_t;

/** Field offsets in the packet control field */
typedef enum {
  CONTROL_OFFSET_READY_FLAG = 0,
  CONTROL_OFFSET_RESERVED0 = 1,
} control_offset_t;

/** Field widths in the packet control field */
typedef enum {
  CONTROL_WIDTH_READY_FLAG = 1,
  CONTROL_WIDTH_RESERVED0 = 31,
} control_width_t;

/** Packet header */
typedef struct {
  /** Tagged pointer to the next packet in an intrusive stack */
  uint64_t next;
  /** Bitmask that represents payload slots with valid data */
  uint64_t activemask;
  /** Service ID requested by the wave */
  uint32_t service;
  /** Control bits.
   *  \li \c READY flag is bit 0. Indicates packet awaiting a host response.
   */
  uint32_t control;
} header_t;

/** \brief Hostcall state.
 *
 *  Holds the state of hostcalls being requested by all kernels that
 *  share the same hostcall state. There is usually one buffer per
 *  device queue.
 */
typedef struct {
  /** Array of 2^index_size packet headers */
  header_t *headers;
  /** Array of 2^index_size packet payloads */
  payload_t *payloads;
  /** Signal used by kernels to indicate new work */
  signal_t doorbell;
  /** Stack of free packets */
  uint64_t free_stack;
  /** Stack of ready packets */
  uint64_t ready_stack;
  /** Number of LSBs in the tagged pointer can index into the packet arrays */
  uint32_t index_size;
  /** Device ID */
  uint32_t device_id;
} buffer_t;

enum { SIGNAL_INIT = UINT64_MAX, SIGNAL_DONE = UINT64_MAX - 1 };

static uint32_t get_buffer_alignment() { return alignof(payload_t); }

static uint32_t set_control_field(uint32_t control, uint8_t offset,
                                  uint8_t width, uint32_t value) {
  uint32_t mask = ~(((1 << width) - 1) << offset);
  control &= mask;
  return control | (value << offset);
}

static uint32_t reset_ready_flag(uint32_t control) {
  return set_control_field(control, CONTROL_OFFSET_READY_FLAG,
                           CONTROL_WIDTH_READY_FLAG, 0);
}

static uint64_t get_ptr_index(uint64_t ptr, uint32_t index_size) {
  return ptr & ((1UL << index_size) - 1);
}

static uintptr_t align_to(uintptr_t value, uint32_t alignment) {
  if (value % alignment == 0)
    return value;
  return value - (value % alignment) + alignment;
}

static uintptr_t get_header_start() {
  return align_to(sizeof(buffer_t), alignof(header_t));
}

static uintptr_t get_payload_start(uint32_t num_packets) {
  auto header_start = get_header_start();
  auto header_end = header_start + sizeof(header_t) * num_packets;
  return align_to(header_end, alignof(payload_t));
}

static size_t get_buffer_size(uint32_t num_packets) {
  size_t buffer_size = get_payload_start(num_packets);
  buffer_size += num_packets * sizeof(payload_t);
  return buffer_size;
}
static uint64_t grab_ready_stack(buffer_t *buffer) {
  return __atomic_exchange_n(&buffer->ready_stack, 0,
                             std::memory_order_acquire);
}
static header_t *get_header(buffer_t *buffer, ulong ptr) {
  return buffer->headers + get_ptr_index(ptr, buffer->index_size);
}
static payload_t *get_payload(buffer_t *buffer, ulong ptr) {
  return buffer->payloads + get_ptr_index(ptr, buffer->index_size);
}

static signal_t create_signal() {
  hsa_signal_t hs;
  hsa_status_t status = hsa_signal_create(SIGNAL_INIT, 0, NULL, &hs);
  if (status != HSA_STATUS_SUCCESS)
    return {0};
  return {hs.handle};
}

static hsa_amd_memory_pool_t static_host_memory_pool;
static hsa_amd_memory_pool_t static_device_memory_pools[AMD_MAX_HSA_AGENTS];
static hsa_agent_t static_hsa_agents[AMD_MAX_HSA_AGENTS];

void save_hsa_statics(uint32_t device_id, hsa_amd_memory_pool_t HostMemoryPool,
                      hsa_amd_memory_pool_t DevMemoryPool,
                      hsa_agent_t hsa_agent) {
  assert(device_id < AMD_MAX_HSA_AGENTS && "Supports up n GPUs");
  static_host_memory_pool = HostMemoryPool;
  static_device_memory_pools[device_id] = DevMemoryPool;
  static_hsa_agents[device_id] = hsa_agent;
}

// ====== START of helper functions for execute_service ======
service_rc host_device_mem_free(void *ptr) {
  hsa_status_t err = hsa_amd_memory_pool_free(ptr);
  if (err == HSA_STATUS_SUCCESS)
    return _RC_SUCCESS;
  else
    return _RC_ERROR_MEMFREE;
}

service_rc host_malloc(void **ptr, size_t size, uint32_t devid) {
  hsa_amd_memory_pool_t MemoryPool = static_host_memory_pool;
  hsa_agent_t agent = static_hsa_agents[devid];
  hsa_status_t err = hsa_amd_memory_pool_allocate(MemoryPool, size, 0, ptr);
  if (err == HSA_STATUS_SUCCESS)
    err = hsa_amd_agents_allow_access(1, &agent, NULL, *ptr);
  if (err != HSA_STATUS_SUCCESS)
    thread_abort(_RC_ERROR_HSAFAIL);
  return _RC_SUCCESS;
}

service_rc device_malloc(void **mem, size_t size, uint32_t devid) {
  hsa_amd_memory_pool_t MemoryPool = static_device_memory_pools[devid];
  hsa_status_t err = hsa_amd_memory_pool_allocate(MemoryPool, size, 0, mem);
  if (err != HSA_STATUS_SUCCESS)
    thread_abort(_RC_ERROR_HSAFAIL);
  return _RC_SUCCESS;
}

void thread_abort(service_rc rc) {
  fprintf(stderr, "hostrpc thread_abort called with code %d\n", rc);
  abort();
}
// ====== END helper functions for execute_service ======

/** \brief Locked reference to critical data.
 *
 *         Simpler version of the LockedAccessor in HIP sources.
 *
 *         Protects access to the member _data with a lock acquired on
 *         contruction/destruction. T must contain a _mutex field
 *         which meets the BasicLockable requirements (lock/unlock)
 */
template <typename T> struct locked_accessor_t {
  locked_accessor_t(T &criticalData) : _criticalData(&criticalData) {
    _criticalData->_mutex.lock();
  };
  ~locked_accessor_t() { _criticalData->_mutex.unlock(); }
  // Syntactic sugar so -> can be used to get the underlying type.
  T *operator->() { return _criticalData; };

private:
  T *_criticalData;
};
struct record_t {
  bool discarded;
};
struct critical_data_t {
  std::unordered_map<buffer_t *, record_t> buffers;
  std::mutex _mutex;
};
typedef locked_accessor_t<critical_data_t> locked_critical_data_t;

typedef struct {
  hsa_queue_t *hsa_q;
  buffer_t *hcb;
  uint32_t devid;
} hsaq_buf_entry_t;

extern "C" void handler_SERVICE_SANITIZER(payload_t *packt_payload,
                                          uint64_t activemask,
                                          uint32_t gpu_device,
                                          UriLocator *uri_locator);

static bool static_version_was_checked = false;
struct consumer_t {
private:
  signal_t doorbell;
  std::thread thread;
  critical_data_t critical_data;
  UriLocator *urilocator;
  consumer_t(signal_t _doorbell) : doorbell(_doorbell) {}
  // Table of hsa_q's and their associated buffer_t's
  std::list<hsaq_buf_entry_t *> hsaq_bufs;

public:
  static consumer_t *create_consumer();

  hsaq_buf_entry_t *add_hsaq_buf_entry(buffer_t *hcb, hsa_queue_t *hsa_q,
                                       uint32_t devid) {
    hsaq_buf_entry_t *new_hsaq_buf = new hsaq_buf_entry_t;
    new_hsaq_buf->hcb = hcb;
    new_hsaq_buf->devid = devid;
    new_hsaq_buf->hsa_q = hsa_q;
    hsaq_bufs.push_back(new_hsaq_buf);
    return new_hsaq_buf;
  }

  hsaq_buf_entry_t *find_hsaq_buf_entry(hsa_queue_t *hsa_q) {
    for (auto hsaq_buf : hsaq_bufs) {
      if (hsaq_buf->hsa_q == hsa_q)
        return hsaq_buf;
    }
    return NULL;
  }

  service_rc check_version(uint device_vrm) const {
    if (device_vrm == (unsigned int)HOSTEXEC_VRM)
      return _RC_SUCCESS;
    uint device_version_release = device_vrm >> 6;
    if (device_version_release != HOSTEXEC_VERSION_RELEASE) {
      fprintf(stderr,
              "ERROR Incompatible device and host release\n     Device "
              "release(%d)\n     Host release(%d)\n",
              device_version_release, HOSTEXEC_VERSION_RELEASE);
      return _RC_ERROR_WRONGVERSION;
    }
    if (device_vrm > HOSTEXEC_VRM) {
      fprintf(stderr,
              "ERROR Incompatible device and host version\n      Device "
              "version(%d)\n     Host version(%d)\n",
              device_vrm, HOSTEXEC_VERSION_RELEASE);
      fprintf(stderr,
              "         Upgrade libomptarget runtime on your system.\n");
      return _RC_ERROR_OLDHOSTVERSIONMOD;
    }
    if (device_vrm < HOSTEXEC_VRM) {
      unsigned int host_ver = ((unsigned int)HOSTEXEC_VRM) >> 12;
      unsigned int host_rel = (((unsigned int)HOSTEXEC_VRM) << 20) >> 26;
      unsigned int host_mod = (((unsigned int)HOSTEXEC_VRM) << 26) >> 26;
      unsigned int dev_ver = ((unsigned int)device_vrm) >> 12;
      unsigned int dev_rel = (((unsigned int)device_vrm) << 20) >> 26;
      unsigned int dev_mod = (((unsigned int)device_vrm) << 26) >> 26;
      fprintf(
          stderr,
          "WARNING:  Device mod version < host mod version \n          Device "
          "version: %d.%d.%d\n          Host version:   %d.%d.%d\n",
          dev_ver, dev_rel, dev_mod, host_ver, host_rel, host_mod);
      fprintf(stderr,
              "          Consider rebuild binary with more recent compiler.\n");
    }
    return _RC_SUCCESS;
  }

  void process_packets(buffer_t *buffer, uint64_t ready_stack) const {
    // This function is always called from consume_packets, which owns
    // the lock for the critical data.

    // Each wave can submit at most one packet at a time, and all
    // waves independently push ready packets. The stack of packets at
    // this point cannot contain multiple packets from the same wave,
    // so consuming ready packets in a latest-first order does not
    // affect any wave.
    for (decltype(ready_stack) iter = ready_stack, next = 0; iter;
         iter = next) {

      // Remember the next packet pointer. The current packet will
      // get reused from the free stack after we process it.
      auto header = get_header(buffer, iter);
      next = header->next;

      auto payload = get_payload(buffer, iter);
      uint64_t activemask = header->activemask;

      // split the 32-bit service number into service_id and VRM to be checked
      // if device hostrpc or stubs are ahead of this host runtime.
      uint service_id = (header->service << 16) >> 16;
      if (!static_version_was_checked) {
        uint device_vrm = ((uint)(header->service) >> 16);
        service_rc err = check_version(device_vrm);
        if (err != _RC_SUCCESS)
          thread_abort(err);
        static_version_was_checked = true;
      }

      if (service_id == HOSTEXEC_SID_SANITIZER) {
        handler_SERVICE_SANITIZER(payload, activemask, buffer->device_id,
                                  urilocator);
      } else {
        // Serialize calls to execute_service for each active lane
        // TODO: One could use ffs to skip inactive lanes faster.
        for (uint32_t wi = 0; wi != 64; ++wi) {
          uint64_t flag = activemask & ((uint64_t)1 << wi);
          if (flag == 0)
            continue;
          execute_service(service_id, buffer->device_id, payload->slots[wi]);
        }
      }
      __atomic_store_n(&header->control, reset_ready_flag(header->control),
                       std::memory_order_release);
    }
  }

  // FIXME: This cannot be const because it locks critical data.
  // A lock-free implementaiton might make that possible.
  void consume_packets() {
    /* TODO: The consumer iterates over all registered buffers in an
       unspecified order, and for each buffer, processes packets also
       in an unspecified order. This may need a more efficient
       strategy based on the turnaround time for the services
       requested by all these packets.
     */
    uint64_t signal_value = SIGNAL_INIT;
    uint64_t timeout = 1024 * 1024;

    while (true) {
      hsa_signal_t hs{doorbell.handle};
      signal_value =
          hsa_signal_wait_scacquire(hs, HSA_SIGNAL_CONDITION_NE, signal_value,
                                    timeout, HSA_WAIT_STATE_BLOCKED);
      if (signal_value == SIGNAL_DONE) {
        return;
      }

      locked_critical_data_t data(critical_data);

      for (auto ii = data->buffers.begin(), ie = data->buffers.end(); ii != ie;
           /* don't increment here */) {
        auto record = ii->second;
        if (record.discarded) {
          ii = data->buffers.erase(ii);
          continue;
        }

        buffer_t *buffer = ii->first;
        uint64_t F = grab_ready_stack(buffer);
        if (F)
          process_packets(buffer, F);
        ++ii;
      }
    }
    return;
  }

  service_rc launch_service_thread() {
    if (thread.joinable())
      return _RC_ERROR_CONSUMER_ACTIVE;
    thread = std::thread(&consumer_t::consume_packets, this);
    if (!thread.joinable())
      return _RC_ERROR_CONSUMER_LAUNCH_FAILED;
    return _RC_SUCCESS;
  }

  service_rc terminate_service_thread() {
    if (!thread.joinable())
      return _RC_ERROR_CONSUMER_INACTIVE;
    hsa_signal_t signal = {doorbell.handle};
    hsa_signal_store_screlease(signal, SIGNAL_DONE);
    thread.join();
    return _RC_SUCCESS;
  }

  void register_buffer(void *b) {
    locked_critical_data_t data(critical_data);
    auto buffer = reinterpret_cast<buffer_t *>(b);
    auto &record = data->buffers[buffer];
    record.discarded = false;
    buffer->doorbell = doorbell;
    urilocator = new UriLocator();
  }

  service_rc deregister_buffer(void *b) {
    locked_critical_data_t data(critical_data);
    auto buffer = reinterpret_cast<buffer_t *>(b);
    if (data->buffers.count(buffer) == 0)
      return _RC_ERROR_INVALID_REQUEST;
    auto &record = data->buffers[buffer];
    if (record.discarded)
      return _RC_ERROR_INVALID_REQUEST;
    record.discarded = true;
    return _RC_SUCCESS;
  }

  // destructor triggered by delete static_consumer_ptr in hostrpc_terminate().
  ~consumer_t() {
    for (auto hsaq_buf : hsaq_bufs) {
      if (hsaq_buf) {
        deregister_buffer(hsaq_buf->hcb);
        delete hsaq_buf;
      }
    }
    hsaq_bufs.clear();
    terminate_service_thread();
    delete urilocator;
    critical_data.buffers.clear();
    hsa_signal_t hs{doorbell.handle};
    hsa_signal_destroy(hs);
  }

  buffer_t *create_buffer_t(uint32_t num_packets, uint32_t devid) {
    if (num_packets == 0) {
      fprintf(stderr, "hostrpc create_buffer-t num_packets cannot be zero.\n");
      thread_abort(_RC_ERROR_ZEROPACKETS);
    }
    size_t size = get_buffer_size(num_packets);
    uint32_t align = get_buffer_alignment();
    void *newbuffer = NULL;
    service_rc err = host_malloc(&newbuffer, size + align, devid);
    if (!newbuffer || (err != _RC_SUCCESS)) {
      fprintf(stderr, "hostrpc call to host_malloc failed \n");
      thread_abort(err);
    }

    if ((uintptr_t)newbuffer % get_buffer_alignment() != 0) {
      fprintf(stderr, "ERROR: incorrect alignment \n");
      thread_abort(_RC_ERROR_ALIGNMENT);
    }

    //  Initialize the buffer_t
    buffer_t *hb = (buffer_t *)newbuffer;

    hb->headers = (header_t *)((uint8_t *)hb + get_header_start());
    hb->payloads =
        (payload_t *)((uint8_t *)hb + get_payload_start(num_packets));

    uint32_t index_size = 1;
    if (num_packets > 2)
      index_size = 32 - __builtin_clz(num_packets);
    hb->index_size = index_size;
    hb->headers[0].next = 0;

    uint64_t next = 1UL << index_size;
    for (uint32_t ii = 1; ii != num_packets; ++ii) {
      hb->headers[ii].next = next;
      next = ii;
    }
    hb->free_stack = next;
    hb->ready_stack = 0;
    hb->device_id = devid;
    return hb;
  }

}; // end of class/struct consumer_t

consumer_t *consumer_t::create_consumer() {
  signal_t doorbell = create_signal();
  if (doorbell.handle == 0) {
    return nullptr;
  }
  return new consumer_t(doorbell);
}

// Currently, a single instance of consumer_t is created and saved statically.
// This instance starts a single service thread for ALL devices.
static consumer_t *static_consumer_ptr = NULL;

// This is the main hostrpc function called by the amdgpu plugin when
// launching a kernel on a designated hsa_queue_t. This function should only
// be called if any kernel in the device image requires hostrpc services.
extern "C" unsigned long
hostrpc_assign_buffer(hsa_agent_t agent, hsa_queue_t *this_Q,
                      uint32_t device_id, hsa_amd_memory_pool_t HostMemoryPool,
                      hsa_amd_memory_pool_t DevMemoryPool) {
  // Create and launch the services thread
  if (!static_consumer_ptr) {
    static_consumer_ptr = consumer_t::create_consumer();
    service_rc err = static_consumer_ptr->launch_service_thread();
    if (err != _RC_SUCCESS)
      thread_abort(err);
  }

  // quick return to kernel launch if this hsa q is being reused
  hsaq_buf_entry_t *hsaq_buf = static_consumer_ptr->find_hsaq_buf_entry(this_Q);
  if (hsaq_buf)
    return (unsigned long)hsaq_buf->hcb;

  // Helper functions for execute_service need these hsa values saved
  save_hsa_statics(device_id, HostMemoryPool, DevMemoryPool, agent);

  // Get values needed to determine buffer size
  uint32_t numCu;
  hsa_agent_get_info(
      agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &numCu);
  // ErrorCheck(Could not get number of cus, err);
  uint32_t waverPerCu;
  hsa_agent_get_info(agent,
                     (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
                     &waverPerCu);
  // ErrorCheck(Could not get number of waves per cu, err);
  unsigned int minpackets = numCu * waverPerCu;

  //  Create and initialize the new buffer to return to kernel launch
  buffer_t *hcb = static_consumer_ptr->create_buffer_t(minpackets, device_id);

  // Register the buffer for the consumer thread
  static_consumer_ptr->register_buffer(hcb);

  // Cache in hsaq_bufs for reuse
  hsaq_buf = static_consumer_ptr->add_hsaq_buf_entry(hcb, this_Q, device_id);
  return (unsigned long)hcb;
}

extern "C" hsa_status_t hostrpc_terminate() {
  if (static_consumer_ptr) {
    // The consumer_t destructor takes care of all memory returns
    delete static_consumer_ptr;
    static_consumer_ptr = NULL;
  }
  return HSA_STATUS_SUCCESS;
}
