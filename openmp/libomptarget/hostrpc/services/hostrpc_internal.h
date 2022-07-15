#ifndef AMD_HOSTCALL_H
#define AMD_HOSTCALL_H

#include <hsa/hsa.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** \file Support library for invoking host services from the device.
 *
 *  The hostcall consumer defined here is used by the language runtime
 *  to serve requests originating from kernels running on GPU
 *  devices. A typical flow is as follows:
 *
 *  - Create and launch one or more hostcall consumers.
 *  - Create and initialize a hostcall buffer per command queue.
 *  - Register these buffers with the appropriate consumer.
 *  - When a buffer is no longer used, deregister and then free it.
 *  - Destroy the consumer(s) when they are no longer required. Must be
 *    done before exiting the application, so that the consumer
 *    threads can join() correctly.
 *
 *  For a more information, see the accompanying README and the
 *  comments associated with each of the API functions.
 */

// Error codes for service handler functions used in this file
// Some error codes may be returned to device stub functions.
typedef enum hostrpc_status_t {
  HOSTRPC_SUCCESS = 0,
  HOSTRPC_STATUS_UNKNOWN = 1,
  HOSTRPC_STATUS_ERROR = 2,
  HOSTRPC_STATUS_TERMINATE = 3,
  HOSTRPC_DATA_USED_ERROR = 4,
  HOSTRPC_ADDINT_ERROR = 5,
  HOSTRPC_ADDFLOAT_ERROR = 6,
  HOSTRPC_ADDSTRING_ERROR = 7,
  HOSTRPC_UNSUPPORTED_ID_ERROR = 8,
  HOSTRPC_INVALID_ID_ERROR = 9,
  HOSTRPC_ERROR_INVALID_REQUEST = 10,
  HOSTRPC_EXCEED_MAXVARGS_ERROR = 11,
  HOSTRPC_WRONGVERSION_ERROR = 12,
  HOSTRPC_OLDHOSTVERSIONMOD_ERROR = 13,
  HOSTRPC_INVALIDSERVICE_ERROR = 14,
} hostrpc_status_t;
typedef enum {
  AMD_HOSTCALL_SUCCESS,
  AMD_HOSTCALL_ERROR_CONSUMER_ACTIVE,
  AMD_HOSTCALL_ERROR_CONSUMER_INACTIVE,
  AMD_HOSTCALL_ERROR_CONSUMER_LAUNCH_FAILED,
  AMD_HOSTCALL_ERROR_INVALID_REQUEST,
  AMD_HOSTCALL_ERROR_SERVICE_UNKNOWN,
  AMD_HOSTCALL_ERROR_INCORRECT_ALIGNMENT,
  AMD_HOSTCALL_ERROR_NULLPTR
} amd_hostcall_error_t;

void hostrpc_abort(int rc);

const char *amd_hostcall_error_string(amd_hostcall_error_t error);

/// Opaque struct that encapsulates a consumer thread.
typedef struct amd_hostcall_consumer_t amd_hostcall_consumer_t;

/** \brief Create a consumer instance that tracks a single consumer thread.
 *
 *  Each instance manages a unique consumer thread, along with a list
 *  of hostcall buffers that this thread processes. The consumer does
 *  not occupy any resources other than it's own memory footprint
 *  until it is launched.
 *
 *  The corresponding consumer thread must be launched for the
 *  consumer to perform any actual work. The consumer thread can be
 *  launched even without any buffers registered with the
 *  consumer. The API provides thread-safe methods to register buffers
 *  with an active consumer.
 *
 *  A single consumer is sufficient to correctly handle all hostcall
 *  buffers created in the application. The client may safely launch
 *  multiple consumers based on factors external to this library.
 */
amd_hostcall_consumer_t *amd_hostcall_create_consumer(void);

/** \brief Destroy a consumer instance.
 *
 *  If the consumer is active, the corresponding thread is terminated
 *  and join()'ed to the current thread.
 *
 *  Behavious is undefined when called multiple times on the same
 *  pointer, or using a pointer that was not previously created by
 *  amd_hostcall_create_consumer().
 */
void amd_hostcall_destroy_consumer(amd_hostcall_consumer_t *consumer);

/** \brief Determine the buffer size to be allocated for the given
 *         number of packets.
 *
 *  The reported size includes any internal padding required for the
 *  packets and their headers.
 */
size_t amd_hostcall_get_buffer_size(uint32_t num_packets);

/** \brief Alignment required for the start of the buffer.
 */
uint32_t amd_hostcall_get_buffer_alignment(void);

/** \brief Initialize the buffer data-structure.
 *  \param buffer      Pointer to allocated buffer.
 *  \param num_packets Number of packets to be created in the buffer.
 *  \return Error code indicating success or specific failure.
 *
 *  The function assumes that the supplied buffer is sufficiently
 *  large to accomodate the specified number of packets. The value
 *  returned is one of:
 *
 *  \li \c AMD_HOSTCALL_SUCCESS on successful initialization.
 *  \li \c AMD_HOSTCALL_ERROR_NULLPTR if the supplied pointer is NULL.
 *  \li \c AMD_HOSTCALL_ERROR_INCORRECT_ALIGNMENT if the supplied
 *      pointer is not aligned to the value returned by
 *      amd_hostcall_get_buffer_alignment().
 */
amd_hostcall_error_t amd_hostcall_initialize_buffer(void *buffer,
                                                    uint32_t num_packets);

/** \brief Register a buffer with a consumer.
 *
 *  Behaviour is undefined if:
 *  - amd_hostcall_initialize_buffer() was not invoked successfully on
 *    the buffer prior to registration.
 *  - The same buffer is registered with multiple consumers.
 *
 *  The function has no effect if the a buffer is registered multiple
 *  times with the same consumer.
 *
 *  The client must register a buffer before launching any kernel that
 *  accesses that buffer. The client must further ensure that each
 *  buffer is associated with a unique command queue across all
 *  devices.
 */
void amd_hostcall_register_buffer(amd_hostcall_consumer_t *consumer,
                                  void *buffer);

/** \brief Deregister a buffer that is no longer in use.
 *
 *  The client may free this buffer after deregistering it from the
 *  corresponding consumer. Behaviour is undefined if the buffer is
 *  freed without first deregistering it from the consumer.
 *
 *  The value returned is one of:
 *  \li \c AMD_HOSTCALL_SUCCESS on success.
 *  \li \c AMD_HOSTCALL_ERROR_INVALID_REQUEST if the buffer was
 *      previously deregistered or not registered with this consumer.
 */
amd_hostcall_error_t
amd_hostcall_deregister_buffer(amd_hostcall_consumer_t *consumer, void *buffer);

/** \brief Launch the consumer in its own thread.
 *
 *  The value returned is one of:
 *  \li \c AMD_HOSTCALL_SUCCESS on success.
 *  \li \c AMD_HOSTCALL_ERROR_CONSUMER_ACTIVE if the thread is already
 *      running. Such a call has no effect on the consumer thread.
 *  \li \c AMD_HOSTCALL_ERROR_CONSUMER_LAUNCH_FAILED if the thread was
 *      not already running and it failed to launch.
 */
amd_hostcall_error_t
amd_hostcall_launch_consumer(amd_hostcall_consumer_t *consumer);

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

/** \brief Packet payload
 *
 *  Contains 64 slots of 8 ulongs each, one for each workitem in the
 *  wave. A slot with index \c i contains valid data if the
 *  corresponding bit in header_t::activemask is set.
 */
typedef struct {
  uint64_t slots[64][8];
} payload_t;

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

#include "../../plugins/amdgpu/impl/impl_runtime.h"
#include "../plugins/amdgpu/src/utils.h"

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AMD_HOSTCALL_H
