////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2021, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_SUBSET_H_
#define HSA_RUNTIME_SUBSET_H_

typedef struct hsa_dim3_s {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} hsa_dim3_t;

/**
 * @brief Status codes.
 */
typedef enum {
  /**
   * The function has been executed successfully.
   */
  HSA_STATUS_SUCCESS = 0x0,
  /**
   * A traversal over a list of elements has been interrupted by the
   * application before completing.
   */
  HSA_STATUS_INFO_BREAK = 0x1,
  /**
   * A generic error has occurred.
   */
  HSA_STATUS_ERROR = 0x1000,
  /**
   * One of the actual arguments does not meet a precondition stated in the
   * documentation of the corresponding formal argument.
   */
  HSA_STATUS_ERROR_INVALID_ARGUMENT = 0x1001,
  /**
   * The requested queue creation is not valid.
   */
  HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = 0x1002,
  /**
   * The requested allocation is not valid.
   */
  HSA_STATUS_ERROR_INVALID_ALLOCATION = 0x1003,
  /**
   * The agent is invalid.
   */
  HSA_STATUS_ERROR_INVALID_AGENT = 0x1004,
  /**
   * The memory region is invalid.
   */
  HSA_STATUS_ERROR_INVALID_REGION = 0x1005,
  /**
   * The signal is invalid.
   */
  HSA_STATUS_ERROR_INVALID_SIGNAL = 0x1006,
  /**
   * The queue is invalid.
   */
  HSA_STATUS_ERROR_INVALID_QUEUE = 0x1007,
  /**
   * The HSA runtime failed to allocate the necessary resources. This error
   * may also occur when the HSA runtime needs to spawn threads or create
   * internal OS-specific events.
   */
  HSA_STATUS_ERROR_OUT_OF_RESOURCES = 0x1008,
  /**
   * The AQL packet is malformed.
   */
  HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = 0x1009,
  /**
   * An error has been detected while releasing a resource.
   */
  HSA_STATUS_ERROR_RESOURCE_FREE = 0x100A,
  /**
   * An API other than ::hsa_init has been invoked while the reference count
   * of the HSA runtime is 0.
   */
  HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B,
  /**
   * The maximum reference count for the object has been reached.
   */
  HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = 0x100C,
  /**
   * The arguments passed to a functions are not compatible.
   */
  HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = 0x100D,
  /**
   * The index is invalid.
   */
  HSA_STATUS_ERROR_INVALID_INDEX = 0x100E,
  /**
   * The instruction set architecture is invalid.
   */
  HSA_STATUS_ERROR_INVALID_ISA = 0x100F,
  /**
   * The instruction set architecture name is invalid.
   */
  HSA_STATUS_ERROR_INVALID_ISA_NAME = 0x1017,
  /**
   * The code object is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 0x1010,
  /**
   * The executable is invalid.
   */
  HSA_STATUS_ERROR_INVALID_EXECUTABLE = 0x1011,
  /**
   * The executable is frozen.
   */
  HSA_STATUS_ERROR_FROZEN_EXECUTABLE = 0x1012,
  /**
   * There is no symbol with the given name.
   */
  HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = 0x1013,
  /**
   * The variable is already defined.
   */
  HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = 0x1014,
  /**
   * The variable is undefined.
   */
  HSA_STATUS_ERROR_VARIABLE_UNDEFINED = 0x1015,
  /**
   * An HSAIL operation resulted in a hardware exception.
   */
  HSA_STATUS_ERROR_EXCEPTION = 0x1016,
  /**
   * The code object symbol is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = 0x1018,
  /**
   * The executable symbol is invalid.
   */
  HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = 0x1019,
  /**
   * The file descriptor is invalid.
   */
  HSA_STATUS_ERROR_INVALID_FILE = 0x1020,
  /**
   * The code object reader is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = 0x1021,
  /**
   * The cache is invalid.
   */
  HSA_STATUS_ERROR_INVALID_CACHE = 0x1022,
  /**
   * The wavefront is invalid.
   */
  HSA_STATUS_ERROR_INVALID_WAVEFRONT = 0x1023,
  /**
   * The signal group is invalid.
   */
  HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = 0x1024,
  /**
   * The HSA runtime is not in the configuration state.
   */
  HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = 0x1025,
  /**
   * The queue received an error that may require process termination.
   */
  HSA_STATUS_ERROR_FATAL = 0x1026
} hsa_status_t;

/**
 * @brief Agent features.
 */
typedef enum {
  /**
   * The agent supports AQL packets of kernel dispatch type. If this
   * feature is enabled, the agent is also a kernel agent.
   */
  HSA_AGENT_FEATURE_KERNEL_DISPATCH = 1,
  /**
   * The agent supports AQL packets of agent dispatch type.
   */
  HSA_AGENT_FEATURE_AGENT_DISPATCH = 2
} hsa_agent_feature_t;

/**
 * @brief Instruction set architecture.
 */
typedef struct hsa_isa_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_isa_t;

/**
 * @brief Instruction set architecture attributes.
 */
typedef enum {
  /**
   * The length of the ISA name in bytes, not including the NUL terminator. The
   * type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_NAME_LENGTH = 0,
  /**
   * Human-readable description.  The type of this attribute is character array
   * with the length equal to the value of ::HSA_ISA_INFO_NAME_LENGTH attribute.
   */
  HSA_ISA_INFO_NAME = 1,
  /**
   * @deprecated
   *
   * Number of call conventions supported by the instruction set architecture.
   * Must be greater than zero. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_COUNT = 2,
  /**
   * @deprecated
   *
   * Number of work-items in a wavefront for a given call convention. Must be a
   * power of 2 in the range [1,256]. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE = 3,
  /**
   * @deprecated
   *
   * Number of wavefronts per compute unit for a given call convention. In
   * practice, other factors (for example, the amount of group memory used by a
   * work-group) may further limit the number of wavefronts per compute
   * unit. The type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT = 4,
  /**
   * Machine models supported by the instruction set architecture. The type of
   * this attribute is a bool[2]. If the ISA supports the small machine model,
   * the element at index ::HSA_MACHINE_MODEL_SMALL is true. If the ISA supports
   * the large model, the element at index ::HSA_MACHINE_MODEL_LARGE is true.
   */
  HSA_ISA_INFO_MACHINE_MODELS = 5,
  /**
   * Profiles supported by the instruction set architecture. The type of this
   * attribute is a bool[2]. If the ISA supports the base profile, the element
   * at index ::HSA_PROFILE_BASE is true. If the ISA supports the full profile,
   * the element at index ::HSA_PROFILE_FULL is true.
   */
  HSA_ISA_INFO_PROFILES = 6,
  /**
   * Default floating-point rounding modes supported by the instruction set
   * architecture. The type of this attribute is a bool[3]. The value at a given
   * index is true if the corresponding rounding mode in
   * ::hsa_default_float_rounding_mode_t is supported. At least one default mode
   * has to be supported.
   *
   * If the default mode is supported, then
   * ::HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES must report that
   * both the zero and the near roundings modes are supported.
   */
  HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES = 7,
  /**
   * Default floating-point rounding modes supported by the instruction set
   * architecture in the Base profile. The type of this attribute is a
   * bool[3]. The value at a given index is true if the corresponding rounding
   * mode in ::hsa_default_float_rounding_mode_t is supported. The value at
   * index HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT must be false.  At least one
   * of the values at indexes ::HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO or
   * HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR must be true.
   */
  HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 8,
  /**
   * Flag indicating that the f16 HSAIL operation is at least as fast as the
   * f32 operation in the instruction set architecture. The type of this
   * attribute is bool.
   */
  HSA_ISA_INFO_FAST_F16_OPERATION = 9,
  /**
   * Maximum number of work-items of each dimension of a work-group.  Each
   * maximum must be greater than 0. No maximum can exceed the value of
   * ::HSA_ISA_INFO_WORKGROUP_MAX_SIZE. The type of this attribute is
   * uint16_t[3].
   */
  HSA_ISA_INFO_WORKGROUP_MAX_DIM = 12,
  /**
   * Maximum total number of work-items in a work-group. The type
   * of this attribute is uint32_t.
   */
  HSA_ISA_INFO_WORKGROUP_MAX_SIZE = 13,
  /**
   * Maximum number of work-items of each dimension of a grid. Each maximum must
   * be greater than 0, and must not be smaller than the corresponding value in
   * ::HSA_ISA_INFO_WORKGROUP_MAX_DIM. No maximum can exceed the value of
   * ::HSA_ISA_INFO_GRID_MAX_SIZE. The type of this attribute is
   * ::hsa_dim3_t.
   */
  HSA_ISA_INFO_GRID_MAX_DIM = 14,
  /**
   * Maximum total number of work-items in a grid. The type of this
   * attribute is uint64_t.
   */
  HSA_ISA_INFO_GRID_MAX_SIZE = 16,
  /**
   * Maximum number of fbarriers per work-group. Must be at least 32. The
   * type of this attribute is uint32_t.
   */
  HSA_ISA_INFO_FBARRIER_MAX_SIZE = 17
} hsa_isa_info_t;

/**
 * @brief Struct containing an opaque handle to an agent, a device that
 * participates in the HSA memory model. An agent can submit AQL packets for
 * execution, and may also accept AQL packets for execution (agent dispatch
 * packets or kernel dispatch packets launching HSAIL-derived binaries).
 */
typedef struct hsa_agent_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_agent_t;

/**
 * @brief Agent attributes.
 */
typedef enum {
  /**
   * Agent name. The type of this attribute is a NUL-terminated char[64]. The
   * name must be at most 63 characters long (not including the NUL terminator)
   * and all array elements not used for the name must be NUL.
   */
  HSA_AGENT_INFO_NAME = 0,
  /**
   * Name of vendor. The type of this attribute is a NUL-terminated char[64].
   * The name must be at most 63 characters long (not including the NUL
   * terminator) and all array elements not used for the name must be NUL.
   */
  HSA_AGENT_INFO_VENDOR_NAME = 1,
  /**
   * Agent capability. The type of this attribute is ::hsa_agent_feature_t.
   */
  HSA_AGENT_INFO_FEATURE = 2,
  /**
   * @deprecated Query ::HSA_ISA_INFO_MACHINE_MODELS for a given intruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Machine model supported by the agent. The type of this attribute is
   * ::hsa_machine_model_t.
   */
  HSA_AGENT_INFO_MACHINE_MODEL = 3,
  /**
   * @deprecated Query ::HSA_ISA_INFO_PROFILES for a given intruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Profile supported by the agent. The type of this attribute is
   * ::hsa_profile_t.
   */
  HSA_AGENT_INFO_PROFILE = 4,
  /**
   * @deprecated Query ::HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES for a given
   * intruction set architecture supported by the agent instead.  If more than
   * one ISA is supported by the agent, the returned value corresponds to the
   * first ISA enumerated by ::hsa_agent_iterate_isas.
   *
   * Default floating-point rounding mode. The type of this attribute is
   * ::hsa_default_float_rounding_mode_t, but the value
   * ::HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT is not allowed.
   */
  HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5,
  /**
   * @deprecated Query ::HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES
   * for a given intruction set architecture supported by the agent instead.  If
   * more than one ISA is supported by the agent, the returned value corresponds
   * to the first ISA enumerated by ::hsa_agent_iterate_isas.
   *
   * A bit-mask of ::hsa_default_float_rounding_mode_t values, representing the
   * default floating-point rounding modes supported by the agent in the Base
   * profile. The type of this attribute is uint32_t. The default floating-point
   * rounding mode (::HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE) bit must not
   * be set.
   */
  HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 23,
  /**
   * @deprecated Query ::HSA_ISA_INFO_FAST_F16_OPERATION for a given intruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Flag indicating that the f16 HSAIL operation is at least as fast as the
   * f32 operation in the current agent. The value of this attribute is
   * undefined if the agent is not a kernel agent. The type of this
   * attribute is bool.
   */
  HSA_AGENT_INFO_FAST_F16_OPERATION = 24,
  /**
   * @deprecated Query ::HSA_WAVEFRONT_INFO_SIZE for a given wavefront and
   * intruction set architecture supported by the agent instead.  If more than
   * one ISA is supported by the agent, the returned value corresponds to the
   * first ISA enumerated by ::hsa_agent_iterate_isas and the first wavefront
   * enumerated by ::hsa_isa_iterate_wavefronts for that ISA.
   *
   * Number of work-items in a wavefront. Must be a power of 2 in the range
   * [1,256]. The value of this attribute is undefined if the agent is not
   * a kernel agent. The type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_WAVEFRONT_SIZE = 6,
  /**
   * @deprecated Query ::HSA_ISA_INFO_WORKGROUP_MAX_DIM for a given intruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum number of work-items of each dimension of a work-group.  Each
   * maximum must be greater than 0. No maximum can exceed the value of
   * ::HSA_AGENT_INFO_WORKGROUP_MAX_SIZE. The value of this attribute is
   * undefined if the agent is not a kernel agent. The type of this
   * attribute is uint16_t[3].
   */
  HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7,
  /**
   * @deprecated Query ::HSA_ISA_INFO_WORKGROUP_MAX_SIZE for a given intruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum total number of work-items in a work-group. The value of this
   * attribute is undefined if the agent is not a kernel agent. The type
   * of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 8,
  /**
   * @deprecated Query ::HSA_ISA_INFO_GRID_MAX_DIM for a given intruction set
   * architecture supported by the agent instead.
   *
   * Maximum number of work-items of each dimension of a grid. Each maximum must
   * be greater than 0, and must not be smaller than the corresponding value in
   * ::HSA_AGENT_INFO_WORKGROUP_MAX_DIM. No maximum can exceed the value of
   * ::HSA_AGENT_INFO_GRID_MAX_SIZE. The value of this attribute is undefined
   * if the agent is not a kernel agent. The type of this attribute is
   * ::hsa_dim3_t.
   */
  HSA_AGENT_INFO_GRID_MAX_DIM = 9,
  /**
   * @deprecated Query ::HSA_ISA_INFO_GRID_MAX_SIZE for a given intruction set
   * architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum total number of work-items in a grid. The value of this attribute
   * is undefined if the agent is not a kernel agent. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_GRID_MAX_SIZE = 10,
  /**
   * @deprecated Query ::HSA_ISA_INFO_FBARRIER_MAX_SIZE for a given intruction
   * set architecture supported by the agent instead.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Maximum number of fbarriers per work-group. Must be at least 32. The value
   * of this attribute is undefined if the agent is not a kernel agent. The
   * type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_FBARRIER_MAX_SIZE = 11,
  /**
   * @deprecated The maximum number of queues is not statically determined.
   *
   * Maximum number of queues that can be active (created but not destroyed) at
   * one time in the agent. The type of this attribute is uint32_t.
   */
  HSA_AGENT_INFO_QUEUES_MAX = 12,
  /**
   * Minimum number of packets that a queue created in the agent
   * can hold. Must be a power of 2 greater than 0. Must not exceed
   * the value of ::HSA_AGENT_INFO_QUEUE_MAX_SIZE. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13,
  /**
   * Maximum number of packets that a queue created in the agent can
   * hold. Must be a power of 2 greater than 0. The type of this attribute
   * is uint32_t.
   */
  HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14,
  /**
   * Type of a queue created in the agent. The type of this attribute is
   * ::hsa_queue_type32_t.
   */
  HSA_AGENT_INFO_QUEUE_TYPE = 15,
  /**
   * @deprecated NUMA information is not exposed anywhere else in the API.
   *
   * Identifier of the NUMA node associated with the agent. The type of this
   * attribute is uint32_t.
   */
  HSA_AGENT_INFO_NODE = 16,
  /**
   * Type of hardware device associated with the agent. The type of this
   * attribute is ::hsa_device_type_t.
   */
  HSA_AGENT_INFO_DEVICE = 17,
  /**
   * @deprecated Query ::hsa_agent_iterate_caches to retrieve information about
   * the caches present in a given agent.
   *
   * Array of data cache sizes (L1..L4). Each size is expressed in bytes. A size
   * of 0 for a particular level indicates that there is no cache information
   * for that level. The type of this attribute is uint32_t[4].
   */
  HSA_AGENT_INFO_CACHE_SIZE = 18,
  /**
   * @deprecated An agent may support multiple instruction set
   * architectures. See ::hsa_agent_iterate_isas.  If more than one ISA is
   * supported by the agent, the returned value corresponds to the first ISA
   * enumerated by ::hsa_agent_iterate_isas.
   *
   * Instruction set architecture of the agent. The type of this attribute
   * is ::hsa_isa_t.
   */
  HSA_AGENT_INFO_ISA = 19,
  /**
   * Bit-mask indicating which extensions are supported by the agent. An
   * extension with an ID of @p i is supported if the bit at position @p i is
   * set. The type of this attribute is uint8_t[128].
   */
  HSA_AGENT_INFO_EXTENSIONS = 20,
  /**
   * Major version of the HSA runtime specification supported by the
   * agent. The type of this attribute is uint16_t.
   */
  HSA_AGENT_INFO_VERSION_MAJOR = 21,
  /**
   * Minor version of the HSA runtime specification supported by the
   * agent. The type of this attribute is uint16_t.
   */
  HSA_AGENT_INFO_VERSION_MINOR = 22

} hsa_agent_info_t;

/**
 * @brief Agent attributes.
 */
typedef enum hsa_amd_agent_info_s {
  /**
   * Queries UUID of an agent. The value is an Ascii string with a maximum
   * of 21 chars including NUL. The string value consists of two parts: header
   * and body. The header identifies device type (GPU, CPU, DSP) while body
   * encodes UUID as a 16 digit hex string
   *
   * Agents that do not support UUID will return the string "GPU-XX" or
   * "CPU-XX" or "DSP-XX" depending upon their device type ::hsa_device_type_t
   */
  HSA_AMD_AGENT_INFO_UUID = 0xA011
} hsa_amd_agent_info_t;

/**
 * @brief Hardware device type.
 */
typedef enum {
  /**
   * CPU device.
   */
  HSA_DEVICE_TYPE_CPU = 0,
  /**
   * GPU device.
   */
  HSA_DEVICE_TYPE_GPU = 1,
  /**
   * DSP device.
   */
  HSA_DEVICE_TYPE_DSP = 2
} hsa_device_type_t;
#endif
