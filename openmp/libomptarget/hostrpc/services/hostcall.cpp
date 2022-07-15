#include "../plugins/amdgpu/impl/rt.h"
#include "../plugins/amdgpu/src/utils.h"
#include "../src/hostrpc.h"
#include "hostrpc_internal.h"
#include "hsa/hsa.h"
#include "urilocator.h"
#include <assert.h>
#include <atomic>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifndef NDEBUG
bool debug_mode;
#define WHEN_DEBUG(xxx)                                                        \
  do {                                                                         \
    if (debug_mode) {                                                          \
      xxx;                                                                     \
    }                                                                          \
  } while (false)
#else
#define WHEN_DEBUG(xxx)
#endif // NDEBUG

enum { SIGNAL_INIT = UINT64_MAX, SIGNAL_DONE = UINT64_MAX - 1 };

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

static signal_t create_signal() {
  hsa_signal_t hs;
  hsa_status_t status = hsa_signal_create(SIGNAL_INIT, 0, NULL, &hs);
  if (status != HSA_STATUS_SUCCESS)
    return {0};
  return {hs.handle};
}

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

/** \brief Encapsulates the entire consumer thread functionality.
 *
 *  The C API exposed in the header is a thin wrapper around this
 *  class. This ensures that the C++ interface is easy to expose if
 *  required.
 */
struct amd_hostcall_consumer_t {
private:
  signal_t doorbell;
  std::thread thread;
  critical_data_t critical_data;
  UriLocator *urilocator;
  amd_hostcall_consumer_t(signal_t _doorbell) : doorbell(_doorbell) {}

public:
  void register_buffer(void *buffer);
  amd_hostcall_error_t deregister_buffer(void *buffer);

  void process_packets(buffer_t *buffer, uint64_t F) const;
  // FIXME: This cannot be const because it locks critical data. A
  // lock-free implementaiton might make that possible.
  void consume_packets();

  amd_hostcall_error_t launch();
  amd_hostcall_error_t terminate();

  static amd_hostcall_consumer_t *create();
  ~amd_hostcall_consumer_t();
};

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

extern "C" void hostrpc_handler_SERVICE_SANITIZER(payload_t *payload,
                                                  uint64_t activemask,
                                                  const uint32_t *dev_ptr,
                                                  UriLocator *uri_locator);

extern "C" void hostrpc_execute_service(uint32_t service_id,
                                        uint32_t *device_ptr,
                                        uint64_t *payload);

// FIXME: Clean up this diagnostic and die properly
static bool hostrpc_version_checked;

static hostrpc_status_t hostrpc_version_check(unsigned int device_vrm) {
  if (device_vrm == (unsigned int)HOSTRPC_VRM)
    return HOSTRPC_SUCCESS;
  uint device_version_release = device_vrm >> 6;
  if (device_version_release != HOSTRPC_VERSION_RELEASE) {
    printf("ERROR Incompatible device and host release\n      Device "
           "release(%d)\n      Host release(%d)\n",
           device_version_release, HOSTRPC_VERSION_RELEASE);
    return HOSTRPC_WRONGVERSION_ERROR;
  }
  if (device_vrm > HOSTRPC_VRM) {
    printf("ERROR Incompatible device and host version \n       Device "
           "version(%d)\n      Host version(%d)\n",
           device_vrm, HOSTRPC_VERSION_RELEASE);
    printf("          Upgrade libomptarget runtime on your system.\n");
    return HOSTRPC_OLDHOSTVERSIONMOD_ERROR;
  }
  if (device_vrm < HOSTRPC_VRM) {
    unsigned int host_ver = ((unsigned int)HOSTRPC_VRM) >> 12;
    unsigned int host_rel = (((unsigned int)HOSTRPC_VRM) << 20) >> 26;
    unsigned int host_mod = (((unsigned int)HOSTRPC_VRM) << 26) >> 26;
    unsigned int dev_ver = ((unsigned int)device_vrm) >> 12;
    unsigned int dev_rel = (((unsigned int)device_vrm) << 20) >> 26;
    unsigned int dev_mod = (((unsigned int)device_vrm) << 26) >> 26;
    printf("WARNING:  Device mod version < host mod version \n          Device "
           "version: %d.%d.%d\n          Host version:   %d.%d.%d\n",
           dev_ver, dev_rel, dev_mod, host_ver, host_rel, host_mod);
    printf("          Consider rebuild binary with more recent compiler.\n");
  }
  return HOSTRPC_SUCCESS;
}

void amd_hostcall_consumer_t::process_packets(buffer_t *buffer,
                                              uint64_t ready_stack) const {
  // This function is always called from consume_packets, which owns
  // the lock for the critical data.

  WHEN_DEBUG(std::cout << "process packets starting with " << ready_stack
                       << std::endl);

  // Each wave can submit at most one packet at a time, and all
  // waves independently push ready packets. The stack of packets at
  // this point cannot contain multiple packets from the same wave,
  // so consuming ready packets in a latest-first order does not
  // affect any wave.
  for (decltype(ready_stack) iter = ready_stack, next = 0; iter; iter = next) {
    WHEN_DEBUG(std::cout << "processing ptr: " << iter << std::endl);
    WHEN_DEBUG(std::cout << "packet index: " << std::dec
                         << get_ptr_index(iter, buffer->index_size)
                         << std::endl);

    // Remember the next packet pointer. The current packet will
    // get reused from the free stack after we process it.
    auto header = get_header(buffer, iter);
    next = header->next;

    WHEN_DEBUG(std::cout << "packet service: " << (uint32_t)header->service
                         << std::endl);

    auto payload = get_payload(buffer, iter);
    uint64_t activemask = header->activemask;
    WHEN_DEBUG(std::cout << "activemask: " << std::hex << activemask
                         << std::endl);

    // split the 32-bit service number into service_id and VRM to be checked
    // if device hostrpc or stubs are ahead of this host runtime.
    uint service_id = (header->service << 16) >> 16;
    if (!hostrpc_version_checked) {
      uint device_vrm = ((uint)(header->service) >> 16);
      hostrpc_status_t err = hostrpc_version_check(device_vrm);
      if (err != HOSTRPC_SUCCESS)
        hostrpc_abort(err);
      hostrpc_version_checked = true;
    }

    if (service_id == HOSTRPC_SERVICE_SANITIZER) {
      hostrpc_handler_SERVICE_SANITIZER(payload, activemask,
                                        &(buffer->device_id), urilocator);
    } else {
      // TODO: One could use ffs to skip inactive lanes faster.
      for (uint32_t wi = 0; wi != 64; ++wi) {
        uint64_t flag = activemask & ((uint64_t)1 << wi);
        if (flag == 0)
          continue;
        uint64_t *slot = payload->slots[wi];
        hostrpc_execute_service(service_id, &(buffer->device_id), slot);
      }
    }
    __atomic_store_n(&header->control, reset_ready_flag(header->control),
                     std::memory_order_release);
  }
}

void amd_hostcall_consumer_t::consume_packets() {
  /* TODO: The consumer iterates over all registered buffers in an
     unspecified order, and for each buffer, processes packets also
     in an unspecified order. This may need a more efficient
     strategy based on the turnaround time for the services
     requested by all these packets.
   */
  WHEN_DEBUG(std::cout << "launched consumer" << std::endl);
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
      WHEN_DEBUG(std::cout << "grabbed ready stack: " << F << std::endl);
      if (F) {
        process_packets(buffer, F);
      }
      ++ii;
    }
  }

  return;
}

amd_hostcall_error_t amd_hostcall_consumer_t::launch() {
  if (thread.joinable())
    return AMD_HOSTCALL_ERROR_CONSUMER_ACTIVE;
  thread = std::thread(&amd_hostcall_consumer_t::consume_packets, this);
  if (!thread.joinable())
    return AMD_HOSTCALL_ERROR_CONSUMER_LAUNCH_FAILED;

  return AMD_HOSTCALL_SUCCESS;
}

amd_hostcall_error_t amd_hostcall_consumer_t::terminate() {
  if (!thread.joinable())
    return AMD_HOSTCALL_ERROR_CONSUMER_INACTIVE;
  hsa_signal_t signal = {doorbell.handle};
  hsa_signal_store_screlease(signal, SIGNAL_DONE);
  thread.join();
  return AMD_HOSTCALL_SUCCESS;
}

void amd_hostcall_consumer_t::register_buffer(void *b) {
  locked_critical_data_t data(critical_data);
  auto buffer = reinterpret_cast<buffer_t *>(b);
  auto &record = data->buffers[buffer];
  WHEN_DEBUG(std::cout << "registered buffer: " << std::hex << b << std::endl);
  record.discarded = false;
  buffer->doorbell = doorbell;
  urilocator = new UriLocator();
  WHEN_DEBUG(std::cout << "signal: " << buffer->doorbell.handle << std::endl);
}

amd_hostcall_error_t amd_hostcall_consumer_t::deregister_buffer(void *b) {
  locked_critical_data_t data(critical_data);
  auto buffer = reinterpret_cast<buffer_t *>(b);
  if (data->buffers.count(buffer) == 0)
    return AMD_HOSTCALL_ERROR_INVALID_REQUEST;
  auto &record = data->buffers[buffer];
  if (record.discarded)
    return AMD_HOSTCALL_ERROR_INVALID_REQUEST;
  record.discarded = true;
  return AMD_HOSTCALL_SUCCESS;
}

amd_hostcall_consumer_t::~amd_hostcall_consumer_t() {
  terminate();
  delete urilocator;
  critical_data.buffers.clear();
  hsa_signal_t hs{doorbell.handle};
  hsa_signal_destroy(hs);
}

amd_hostcall_consumer_t *amd_hostcall_consumer_t::create() {
  signal_t doorbell = create_signal();
  if (doorbell.handle == 0) {
    return nullptr;
  }
  return new amd_hostcall_consumer_t(doorbell);
}

amd_hostcall_consumer_t *amd_hostcall_create_consumer() {
  return amd_hostcall_consumer_t::create();
}

size_t amd_hostcall_get_buffer_size(uint32_t num_packets) {
  WHEN_DEBUG(std::cout << "header start: " << get_header_start() << std::endl);
  WHEN_DEBUG(std::cout << "payload start: " << get_payload_start(num_packets)
                       << std::endl);
  size_t buffer_size = get_payload_start(num_packets);
  buffer_size += num_packets * sizeof(payload_t);
  return buffer_size;
}

uint32_t amd_hostcall_get_buffer_alignment() { return alignof(payload_t); }

amd_hostcall_error_t amd_hostcall_initialize_buffer(void *buffer,
                                                    uint32_t num_packets) {
  if (!buffer) {
    return AMD_HOSTCALL_ERROR_NULLPTR;
  }

  if ((uintptr_t)buffer % amd_hostcall_get_buffer_alignment() != 0) {
    return AMD_HOSTCALL_ERROR_INCORRECT_ALIGNMENT;
  }

  buffer_t *hb = (buffer_t *)buffer;

  hb->headers = (header_t *)((uint8_t *)hb + get_header_start());
  hb->payloads = (payload_t *)((uint8_t *)hb + get_payload_start(num_packets));

  uint32_t index_size = 1;
  if (num_packets > 2)
    index_size = 32 - __builtin_clz(num_packets);
  WHEN_DEBUG(std::cout << "index size: " << index_size << std::endl);
  hb->index_size = index_size;
  hb->headers[0].next = 0;

  uint64_t next = 1UL << index_size;
  for (uint32_t ii = 1; ii != num_packets; ++ii) {
    hb->headers[ii].next = next;
    next = ii;
  }
  hb->free_stack = next;
  hb->ready_stack = 0;

  return AMD_HOSTCALL_SUCCESS;
}

void amd_hostcall_register_buffer(amd_hostcall_consumer_t *consumer,
                                  void *buffer) {
  consumer->register_buffer(buffer);
}

amd_hostcall_error_t
amd_hostcall_deregister_buffer(amd_hostcall_consumer_t *consumer,
                               void *buffer) {
  return consumer->deregister_buffer(buffer);
}

amd_hostcall_error_t
amd_hostcall_launch_consumer(amd_hostcall_consumer_t *consumer) {
  return consumer->launch();
}

void amd_hostcall_destroy_consumer(amd_hostcall_consumer_t *consumer) {
  delete consumer;
}

void amd_hostcall_enable_debug() {
#ifndef NDEBUG
  debug_mode = true;
#endif
}

const char *amd_hostcall_error_string(amd_hostcall_error_t error) {
  switch (error) {
  case AMD_HOSTCALL_SUCCESS:
    return "AMD_HOSTCALL_SUCCESS";
  case AMD_HOSTCALL_ERROR_CONSUMER_ACTIVE:
    return "AMD_HOSTCALL_ERROR_CONSUMER_ACTIVE";
  case AMD_HOSTCALL_ERROR_CONSUMER_INACTIVE:
    return "AMD_HOSTCALL_ERROR_CONSUMER_INACTIVE";
  case AMD_HOSTCALL_ERROR_CONSUMER_LAUNCH_FAILED:
    return "AMD_HOSTCALL_ERROR_CONSUMER_LAUNCH_FAILED";
  case AMD_HOSTCALL_ERROR_INVALID_REQUEST:
    return "AMD_HOSTCALL_ERROR_INVALID_REQUEST";
  case AMD_HOSTCALL_ERROR_SERVICE_UNKNOWN:
    return "AMD_HOSTCALL_ERROR_SERVICE_UNKNOWN";
  case AMD_HOSTCALL_ERROR_INCORRECT_ALIGNMENT:
    return "AMD_HOSTCALL_ERROR_INCORRECT_ALIGNMENT";
  case AMD_HOSTCALL_ERROR_NULLPTR:
    return "AMD_HOSTCALL_ERROR_NULLPTR";
  default:
    return "AMD_HOSTCALL_ERROR_UNKNOWN";
  }
}
