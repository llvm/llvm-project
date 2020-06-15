#ifndef AMD_HOSTCALL_IMPL_H
#define AMD_HOSTCALL_IMPL_H

#include "amd_hostcall.h"

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
} buffer_t;

#endif // AMD_HOSTCALL_IMPL_H
