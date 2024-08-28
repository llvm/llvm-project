//===---- execute_service.cpp - support for hostrpc services --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains architecture independed host code for hostrpc services
// Calls to hostrpc_execute are serialized by the thread and packet manager.
// For printf and fprintf, this code reconstructs the host variable argument
// ABI to support alls to vprintf and vfprintf.  This is facilitated by
// a robust buffer packaging scheme defined in Clang codegen. The same buffer
// packaging scheme is used for hostexec fuctions.  Hostexec functions support
// the launching of host variadic functions from the GPU.
//
//===----------------------------------------------------------------------===//

/* MIT License

Copyright © 2023 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include "execute_service.h"
#include "../src/hostexec_internal.h"
#include <assert.h>
#include <cstring>
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple>

// MAXVARGS applies to non-printf vargs functions only.
#define MAXVARGS 32
// NUMFPREGS and FPREGSZ are part of x86 vargs ABI that
// is recreated with the printf support.
#define NUMFPREGS 8
#define FPREGSZ 16

typedef int uint128_t __attribute__((mode(TI)));
struct hostrpc_pfIntRegs {
  uint64_t rdi, rsi, rdx, rcx, r8, r9;
};
typedef struct hostrpc_pfIntRegs hostrpc_pfIntRegs_t; // size = 48 bytes

struct hostrpc_pfRegSaveArea {
  hostrpc_pfIntRegs_t iregs;
  uint128_t freg[NUMFPREGS];
};
typedef struct hostrpc_pfRegSaveArea
    hostrpc_pfRegSaveArea_t; // size = 304 bytes

struct hostrpc_ValistExt {
  uint32_t gp_offset;      /* offset to next available gpr in reg_save_area */
  uint32_t fp_offset;      /* offset to next available fpr in reg_save_area */
  void *overflow_arg_area; /* args that are passed on the stack */
  hostrpc_pfRegSaveArea_t *reg_save_area; /* int and fp registers */
  size_t overflow_size;
} __attribute__((packed));
typedef struct hostrpc_ValistExt hostrpc_ValistExt_t;

/// Prototype for host fallback functions
// typedef uint32_t hostexec_uint_t(void *, ...);
// typedef uint64_t hostexec_uint64_t(void *, ...);
// typedef double   hostexec_double_t(void *, ...);

static service_rc hostrpc_printf(char *buf, size_t bufsz, uint32_t *rc);
static service_rc hostrpc_fprintf(char *buf, size_t bufsz, uint32_t *rc);

template <typename T, typename FT>
static service_rc hostexec_service(char *buf, size_t bufsz, T *rc);

static void handler_SERVICE_PRINTF(uint32_t device_id, uint64_t *payload) {
  size_t bufsz = (size_t)payload[0];
  char *device_buffer = (char *)payload[1];
  uint uint_value;
  service_rc rc = hostrpc_printf(device_buffer, bufsz, &uint_value);
  payload[0] = (uint64_t)uint_value; // what the printf returns
  payload[1] = (uint64_t)rc;         // Any errors in the service function
  service_rc rcmem = host_device_mem_free(device_buffer);
  payload[2] = (uint64_t)rcmem;
}
static void handler_SERVICE_FPRINTF(uint32_t device_id, uint64_t *payload) {
  size_t bufsz = (size_t)payload[0];
  char *device_buffer = (char *)payload[1];
  uint uint_value;
  service_rc rc = hostrpc_fprintf(device_buffer, bufsz, &uint_value);
  payload[0] = (uint64_t)uint_value; // what the printf returns
  payload[1] = (uint64_t)rc;         // Any errors in the service function
  service_rc err = host_device_mem_free(device_buffer);
  payload[2] = (uint64_t)err;
}

template <typename T, typename TF>
static void handler_SERVICE_VARFN(uint32_t device_id, uint64_t *payload) {
  size_t bufsz = (size_t)payload[0];
  char *device_buffer = (char *)payload[1];
  payload[0] = 0; // zero 64 bits
  service_rc rc = hostexec_service<T, TF>(device_buffer, bufsz, (T *)payload);
  // payload[0] has the return value
  // payload[1] reserved for 128 bit values such as double complex
  // payload[2-3] reserved for 256 bit return values
  payload[1] = (uint64_t)rc; // any errors in the service function
  service_rc err = host_device_mem_free(device_buffer);
  payload[2] = (uint64_t)err; // any errors on memory free
}

static void handler_SERVICE_HOST_MALLOC(uint32_t device_id, uint64_t *payload) {
  void *ptr = NULL;
  // CPU device ID 0 is the fine grain memory
  size_t sz = (size_t)payload[0];
  service_rc err = host_malloc(&ptr, sz, device_id);
  payload[0] = (uint64_t)err;
  payload[1] = (uint64_t)ptr;
}

//  SERVICE_MALLOC & SERVICE_FREE are for allocating a heap of device memory
//  only used by the device to be used for device side malloc and free.
//  This is called by __ockl_devmem_request. For allocating memory visible
//  to both host and device user SERVICE_HOST_MALLOC. The corresponding
//  vargs function will release this
static void handler_SERVICE_MALLOC(uint32_t device_id, uint64_t *payload) {
  void *ptr = NULL;
  size_t sz = (size_t)payload[0];
  service_rc err = device_malloc(&ptr, sz, device_id);
  payload[0] = (uint64_t)err;
  payload[1] = (uint64_t)ptr;
}

#if 0
void fort_ptr_assign_i8(void *arg0, void *arg1, void *arg2, void *arg3, void *arg4) {
  printf("\n\n ERROR: hostrpc service FTNASSIGN is not functional\n\n");
};
service_rc FtnAssignWrapper(void *arg0, void *arg1, void *arg2, void *arg3, void *arg4) {
  fort_ptr_assign_i8(arg0, arg1, arg2, arg3, arg4);
  return HSA_STATUS_SUCCESS;
}

service_rc ftn_assign_wrapper(void *arg0, void *arg1, void *arg2, void *arg3,
                                void *arg4) {
  return FtnAssignWrapper(arg0, arg1, arg2, arg3, arg4);
}

static void handler_SERVICE_FTNASSIGN(uint32_t device_id,
                                              uint64_t *payload) {
  void *ptr = NULL;
  service_rc err = ftn_assign_wrapper((void *)payload[0], (void *)payload[1],
                                        (void *)payload[2], (void *)payload[3],
                                        (void *)payload[4]);
  payload[0] = (uint64_t)err;
  payload[1] = (uint64_t)ptr;
}
#endif

static void handler_SERVICE_FREE(uint32_t device_id, uint64_t *payload) {
  char *device_buffer = (char *)payload[0];
  service_rc err = host_device_mem_free(device_buffer);
  payload[0] = (uint64_t)err;
}

static bool trace_init = false;
static bool host_exec_trace;
static char* TrcStrs[HOSTEXEC_SID_VOID+1] = {"unsed", "terminate", "device_malloc",
        "host_malloc", "free", "printf", "fprintf", "ftnassign", "sanatizer",
        "uint", "uint64", "double", "int", "long", "float" , "void"};
// The consumer thread will serialize each active lane and call execute_service
// for the service request. These services are intended to be architecturally
// independent.
void execute_service(uint32_t service_id, uint32_t device_id,
                     uint64_t *payload) {
  if (!trace_init) {
    trace_init = true;
    if (char *EnvStr = getenv("LIBOMPTARGET_HOSTEXEC_TRACE"))
      host_exec_trace = std::atoi(EnvStr) != 0;
  }
  if (host_exec_trace)
    fprintf(stderr, "Hostexec service: %s SrvId: %d DevId: %d PayLoad: %lu\n",
                    TrcStrs[service_id], service_id, device_id, payload[0]);

  switch (service_id) {
  case HOSTEXEC_SID_PRINTF:
    handler_SERVICE_PRINTF(device_id, payload);
    break;
  case HOSTEXEC_SID_FPRINTF:
    handler_SERVICE_FPRINTF(device_id, payload);
    break;
  case HOSTEXEC_SID_VOID:
    // Cannot return a void in template so just use uint64_t
    handler_SERVICE_VARFN<uint64_t, hostexec_uint64_t>(device_id, payload);
    break;
  case HOSTEXEC_SID_UINT:
    handler_SERVICE_VARFN<uint, hostexec_uint_t>(device_id, payload);
    break;
  case HOSTEXEC_SID_UINT64:
    handler_SERVICE_VARFN<uint64_t, hostexec_uint64_t>(device_id, payload);
    break;
  case HOSTEXEC_SID_DOUBLE:
    handler_SERVICE_VARFN<double, hostexec_double_t>(device_id, payload);
    break;
  case HOSTEXEC_SID_INT:
    handler_SERVICE_VARFN<int, hostexec_int_t>(device_id, payload);
    break;
  case HOSTEXEC_SID_LONG:
    handler_SERVICE_VARFN<long, hostexec_long_t>(device_id, payload);
    break;
  case HOSTEXEC_SID_FLOAT:
    handler_SERVICE_VARFN<float, hostexec_float_t>(device_id, payload);
    break;
  case HOSTEXEC_SID_HOST_MALLOC:
    handler_SERVICE_HOST_MALLOC(device_id, payload);
    break;
  case HOSTEXEC_SID_DEVICE_MALLOC:
    handler_SERVICE_MALLOC(device_id, payload);
    break;
    //  case HOSTEXEC_SID_FTNASSIGN:
    //    handler_SERVICE_FTNASSIGN(device_id, payload);
    //    break;
  case HOSTEXEC_SID_FREE:
    handler_SERVICE_FREE(device_id, payload);
    break;
  default:
    fprintf(stderr, "ERROR: hostrpc got a bad service id:%d\n", service_id);
    thread_abort(_RC_INVALIDSERVICE_ERROR);
  }
}

// Support for hostrpc_printf service

// Handle overflow when building the va_list for vprintf
static service_rc hostrpc_pfGetOverflow(hostrpc_ValistExt_t *valist,
                                        size_t needsize) {
  if (needsize < valist->overflow_size)
    return _RC_SUCCESS;

  // Make the overflow area bigger
  size_t stacksize;
  void *newstack;
  if (valist->overflow_size == 0) {
    // Make initial save area big to reduce mallocs
    stacksize = (FPREGSZ * NUMFPREGS) * 2;
    if (needsize > stacksize)
      stacksize = needsize; // maybe a big string
  } else {
    // Initial save area not big enough, double it
    stacksize = valist->overflow_size * 2;
  }
  if (!(newstack = malloc(stacksize))) {
    return _RC_STATUS_ERROR;
  }
  memset(newstack, 0, stacksize);
  if (valist->overflow_size) {
    memcpy(newstack, valist->overflow_arg_area, valist->overflow_size);
    free(valist->overflow_arg_area);
  }
  valist->overflow_arg_area = newstack;
  valist->overflow_size = stacksize;
  return _RC_SUCCESS;
}

// Add an integer to the va_list for vprintf
static service_rc hostrpc_pfAddInteger(hostrpc_ValistExt_t *valist, char *val,
                                       size_t valsize, size_t *stacksize) {
  uint64_t ival;
  switch (valsize) {
  case 1:
    ival = *(uint8_t *)val;
    break;
  case 2:
    ival = *(uint32_t *)val;
    break;
  case 4:
    ival = (*(uint32_t *)val);
    break;
  case 8:
    ival = *(uint64_t *)val;
    break;
  default: {
    return _RC_STATUS_ERROR;
  }
  }
  //  Always copy 8 bytes, sizeof(ival)
  if ((valist->gp_offset + sizeof(ival)) <= sizeof(hostrpc_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), &ival,
           sizeof(ival));
    valist->gp_offset += sizeof(ival);
    return _RC_SUCCESS;
  }
  // Ensure valist overflow area is big enough
  size_t needsize = (size_t)*stacksize + sizeof(ival);
  if (hostrpc_pfGetOverflow(valist, needsize) != _RC_SUCCESS)
    return _RC_STATUS_ERROR;
  // Copy to overflow
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &ival,
         sizeof(ival));

  *stacksize += sizeof(ival);
  return _RC_SUCCESS;
}

// Add a String argument when building va_list for vprintf
static service_rc hostrpc_pfAddString(hostrpc_ValistExt_t *valist, char *val,
                                      size_t strsz, size_t *stacksize) {
  size_t valsize =
      sizeof(char *); // ABI captures pointer to string,  not string
  if ((valist->gp_offset + valsize) <= sizeof(hostrpc_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), val, valsize);
    valist->gp_offset += valsize;
    return _RC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + valsize;
  if (hostrpc_pfGetOverflow(valist, needsize) != _RC_SUCCESS)
    return _RC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, val,
         valsize);
  *stacksize += valsize;
  return _RC_SUCCESS;
}

// Add a floating point value when building va_list for vprintf
static service_rc hostrpc_pfAddFloat(hostrpc_ValistExt_t *valist, char *numdata,
                                     size_t valsize, size_t *stacksize) {
  // FIXME, we can used load because doubles are now aligned
  double dval;
  if (valsize == 4) {
    float fval;
    memcpy(&fval, numdata, 4);
    dval = (double)fval; // Extend single to double per abi
  } else if (valsize == 8) {
    memcpy(&dval, numdata, 8);
  } else {
    return _RC_STATUS_ERROR;
  }
  if ((valist->fp_offset + FPREGSZ) <= sizeof(hostrpc_pfRegSaveArea_t)) {
    memcpy(((char *)valist->reg_save_area + (size_t)(valist->fp_offset)), &dval,
           sizeof(double));
    valist->fp_offset += FPREGSZ;
    return _RC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + sizeof(double);
  if (hostrpc_pfGetOverflow(valist, needsize) != _RC_SUCCESS)
    return _RC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &dval,
         sizeof(double));
  // move only by the size of the double (8 bytes)
  *stacksize += sizeof(double);
  return _RC_SUCCESS;
}

// We would like to get llvm typeID enum from Type.h. e.g.
// #include "../../../../../llvm/include/llvm/IR/Type.h"
// But we cannot include LLVM headers in a runtime function.
// So we a have a manual copy of llvm TypeID enum from Type.h
enum TypeID {
  // PrimitiveTypes
  HalfTyID = 0,  ///< 16-bit floating point type
  BFloatTyID,    ///< 16-bit floating point type (7-bit significand)
  FloatTyID,     ///< 32-bit floating point type
  DoubleTyID,    ///< 64-bit floating point type
  X86_FP80TyID,  ///< 80-bit floating point type (X87)
  FP128TyID,     ///< 128-bit floating point type (112-bit significand)
  PPC_FP128TyID, ///< 128-bit floating point type (two 64-bits, PowerPC)
  VoidTyID,      ///< type with no size
  LabelTyID,     ///< Labels
  MetadataTyID,  ///< Metadata
  X86_MMXTyID,   ///< MMX vectors (64 bits, X86 specific)
  X86_AMXTyID,   ///< AMX vectors (8192 bits, X86 specific)
  TokenTyID,     ///< Tokens

  // Derived types... see DerivedTypes.h file.
  IntegerTyID,        ///< Arbitrary bit width integers
  FunctionTyID,       ///< Functions
  PointerTyID,        ///< Pointers
  StructTyID,         ///< Structures
  ArrayTyID,          ///< Arrays
  FixedVectorTyID,    ///< Fixed width SIMD vector type
  ScalableVectorTyID, ///< Scalable SIMD vector type
  TypedPointerTyID,   ///< Typed pointer used by some GPU targets
  TargetExtTyID,      ///< Target extension type
};

// Build an extended va_list for vprintf by unpacking the buffer
static service_rc hostrpc_pfBuildValist(hostrpc_ValistExt_t *valist,
                                        int NumArgs, char *keyptr,
                                        char *dataptr, char *strptr,
                                        size_t *data_not_used) {
  hostrpc_pfRegSaveArea_t *regs;
  size_t regs_size = sizeof(*regs);
  regs = (hostrpc_pfRegSaveArea_t *)malloc(regs_size);
  if (!regs)
    return _RC_STATUS_ERROR;
  memset(regs, 0, regs_size);
  *valist = (hostrpc_ValistExt_t){
      .gp_offset = 0,
      .fp_offset = 0,
      .overflow_arg_area = NULL,
      .reg_save_area = regs,
      .overflow_size = 0,
  };

  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;

  size_t stacksize = 0;

  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int llvmID = key >> 16;
    unsigned int numbits = (key << 16) >> 16;

    switch (llvmID) {
    case FloatTyID:  ///<  2: 32-bit floating point type
    case DoubleTyID: ///<  3: 64-bit floating point type
    case FP128TyID:  ///<  5: 128-bit floating point type (112-bit mantissa)
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _RC_DATA_USED_ERROR;
      if (valist->fp_offset == 0)
        valist->fp_offset = sizeof(hostrpc_pfIntRegs_t);
      if (hostrpc_pfAddFloat(valist, dataptr, num_bytes, &stacksize))
        return _RC_ADDFLOAT_ERROR;
      break;

    case IntegerTyID: ///< 11: Arbitrary bit width integers
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _RC_DATA_USED_ERROR;
      if (hostrpc_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
        return _RC_ADDINT_ERROR;
      break;

    case PointerTyID:     ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t) * (unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _RC_DATA_USED_ERROR;
        if (hostrpc_pfAddString(valist, (char *)&strptr, strsz, &stacksize))
          return _RC_ADDSTRING_ERROR;
      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return _RC_DATA_USED_ERROR;
        if (hostrpc_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
          return _RC_ADDINT_ERROR;
      }
      break;

    case HalfTyID:           ///<  1: 16-bit floating point type
    case ArrayTyID:          ///< 14: Arrays
    case StructTyID:         ///< 13: Structures
    case FunctionTyID:       ///< 12: Functions
    case TokenTyID:          ///< 10: Tokens
    case X86_MMXTyID:        ///<  9: MMX vectors (64 bits, X86 specific)
    case MetadataTyID:       ///<  8: Metadata
    case LabelTyID:          ///<  7: Labels
    case PPC_FP128TyID:      ///<  6: 128-bit floating point type (two 64-bits,
                             ///<  PowerPC)
    case X86_FP80TyID:       ///<  4: 80-bit floating point type (X87)
    case FixedVectorTyID:    ///< 16: Fixed width SIMD vector type
    case ScalableVectorTyID: ///< 17: Scalable SIMD vector type
    case TypedPointerTyID:   ///< Typed pointer used by some GPU targets
    case TargetExtTyID:      ///< Target extension type
    case VoidTyID:
      return _RC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return _RC_INVALID_ID_ERROR;
    }

    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
  }
  return _RC_SUCCESS;
} // end hostrpc_pfBuildValist

/*
 *  The buffer to pack arguments for all vargs functions has thes 4 sections:
 *  1. Header        datalen 4 bytes
 *                   numargs 4 bytes
 *  2. Keys          A 4-byte key for each arg including string args
 *                   Each 4-byte key contains llvmID and numbits to
 *                   describe the datatype.
 *  3. args_data     Ths data values for each argument.
 *                   Each arg is aligned according to its size.
 *                   If the field is a string
 *                   the dataptr contains the string length.
 *  4. strings_data  Exection time string values
 */
static service_rc hostrpc_fprintf(char *buf, size_t bufsz, uint *rc) {

  // FIXME: Put the collection of these 6 values in a function
  //        All service routines that use vargs will need these values.
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  // Skip past the file pointer and format string argument
  size_t fillerNeeded = ((size_t)dataptr) % 8;
  if (fillerNeeded)
    dataptr += fillerNeeded; // dataptr is now aligned on 8 byte
  // Cannot convert directly to FILE*, so convert to 8-byte size_t first
  FILE *fileptr = (FILE *)*((size_t *)dataptr);
  dataptr += sizeof(FILE *); // skip past file ptr
  NumArgs = NumArgs - 2;
  keyptr += 8; // All keys are 4 bytes
  size_t strsz = (size_t) * (unsigned int *)dataptr;
  dataptr += 4; //  for strings the data value is the size, not a key
  char *fmtstr = strptr;
  strptr += strsz;
  data_not_used -= (sizeof(FILE *) + 4); // 12

  hostrpc_ValistExt_t valist;
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (hostrpc_pfBuildValist(&valist, NumArgs, keyptr, dataptr, strptr,
                            &data_not_used) != _RC_SUCCESS)
    return _RC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(hostrpc_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;

  *rc = vfprintf(fileptr, fmtstr, *real_va_list);

  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);

  return _RC_SUCCESS;
}
//  This the main service routine for printf
static service_rc hostrpc_printf(char *buf, size_t bufsz, uint *rc) {
  if (bufsz == 0)
    return _RC_SUCCESS;

  // Get 6 values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  if (NumArgs <= 0)
    return _RC_ERROR_INVALID_REQUEST;

  // Skip past the format string argument
  char *fmtstr = strptr;
  NumArgs--;
  keyptr += 4;
  size_t strsz = (size_t) * (unsigned int *)dataptr;
  dataptr += 4; // for strings the data value is the size, not a real pointer
  strptr += strsz;
  data_not_used -= 4;

  hostrpc_ValistExt_t valist;
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (hostrpc_pfBuildValist(&valist, NumArgs, keyptr, dataptr, strptr,
                            &data_not_used) != _RC_SUCCESS)
    return _RC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer for
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(hostrpc_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;

  *rc = vprintf(fmtstr, *real_va_list);

  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);

  return _RC_SUCCESS;
}

//---------------- Support for hostexec_* service ---------------------
//

// These are the helper functions for hostexec_<TYPE>_ functions
static uint64_t getuint32(char *val) {
  uint32_t i32 = *(uint32_t *)val;
  return (uint64_t)i32;
}
static uint64_t getuint64(char *val) { return *(uint64_t *)val; }

static void *getfnptr(char *val) {
  uint64_t ival = *(uint64_t *)val;
  return (void *)ival;
}

// build argument array
static service_rc hostrpc_build_vargs_array(int NumArgs, char *keyptr,
                                            char *dataptr, char *strptr,
                                            size_t *data_not_used,
                                            uint64_t *a[MAXVARGS]) {
  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;

  uint argcount = 0;

  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int llvmID = key >> 16;
    unsigned int numbits = (key << 16) >> 16;

    switch (llvmID) {
    case FloatTyID:  ///<  2: 32-bit floating point type
    case DoubleTyID: ///<  3: 64-bit floating point type
    case FP128TyID:  ///<  5: 128-bit floating point type (112-bit mantissa)
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _RC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (uint64_t *)getuint32(dataptr);
      else
        a[argcount] = (uint64_t *)getuint64(dataptr);

      break;

    case IntegerTyID: ///< 11: Arbitrary bit width integers
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _RC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (uint64_t *)getuint32(dataptr);
      else
        a[argcount] = (uint64_t *)getuint64(dataptr);

      break;

    case PointerTyID:     ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t) * (unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _RC_DATA_USED_ERROR;
        a[argcount] = (uint64_t *)((char *)strptr);

      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return _RC_DATA_USED_ERROR;

        a[argcount] = (uint64_t *)getuint64(dataptr);
      }
      break;

    case HalfTyID:           ///<  1: 16-bit floating point type
    case ArrayTyID:          ///< 14: Arrays
    case StructTyID:         ///< 13: Structures
    case FunctionTyID:       ///< 12: Functions
    case TokenTyID:          ///< 10: Tokens
    case X86_MMXTyID:        ///<  9: MMX vectors (64 bits, X86 specific)
    case MetadataTyID:       ///<  8: Metadata
    case LabelTyID:          ///<  7: Labels
    case PPC_FP128TyID:      ///<  6: 128-bit floating point type (two 64-bits,
                             ///<  PowerPC)
    case X86_FP80TyID:       ///<  4: 80-bit floating point type (X87)
    case FixedVectorTyID:    ///< 16: Fixed width SIMD vector type
    case ScalableVectorTyID: ///< 17: Scalable SIMD vector type
    case TypedPointerTyID:   ///< Typed pointer used by some GPU targets
    case TargetExtTyID:      ///< Target extension type
    case VoidTyID:
      return _RC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return _RC_INVALID_ID_ERROR;
    }

    // Move to next argument
    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
    argcount++;
  }
  return _RC_SUCCESS;
}

// Make the vargs function call to the function pointer fnptr
// by casting fnptr to vfnptr. Return uint32_t
template <typename T, typename FT>
static service_rc hostrpc_call_fnptr(uint32_t NumArgs, void *fnptr,
                                     uint64_t *a[MAXVARGS], T *rv) {
  //
  // Currently users are instructed that the first arg must be reserved
  // for device side to store function pointer. Removing this requirement
  // is much more difficult that it appears.  One change of many is to
  // remove fnptr in the call sites below. 2nd is to change the host
  // side macro in hostexec.h to remove the fn arg. This results in the
  // symbol for the variadic function being undefined at GPU link time.
  // This is because device compilation must ignore variadic function
  // definitions.
  //
  // This is a major design decision which would change the test case.
  //
  FT *vfnptr = (FT *)fnptr;

  switch (NumArgs) {
  case 1:
    *rv = vfnptr(fnptr, a[0]);
    break;
  case 2:
    *rv = vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    *rv = vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9]);
    break;
  case 11:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10]);
    break;
  case 12:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11]);
    break;
  case 13:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12]);
    break;
  case 14:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18]);
    break;
  case 20:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19]);
    break;
  case 21:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20]);
    break;
  case 22:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21]);
    break;
  case 23:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
    break;
  case 28:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27]);
    break;
  case 29:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28]);
    break;
  case 30:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29]);
    break;
  case 31:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30]);
    break;
  case 32:
    *rv = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    return _RC_EXCEED_MAXVARGS_ERROR;
  }
  return _RC_SUCCESS;
}

template <typename T, typename FT>
static service_rc hostexec_service(char *buf, size_t bufsz, T *return_value) {
  if (bufsz == 0)
    return _RC_SUCCESS;

  // Get 6 values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  // skip the function pointer arg including any align buffer
  if (((size_t)dataptr) % (size_t)8) {
    dataptr += 4;
    data_not_used -= 4;
  }
  void *fnptr = getfnptr(dataptr);
  NumArgs--;
  keyptr += 4;
  dataptr += 8;
  data_not_used -= 4;

  if (NumArgs <= 0)
    return _RC_ERROR_INVALID_REQUEST;

  uint64_t *a[MAXVARGS];
  if (hostrpc_build_vargs_array(NumArgs, keyptr, dataptr, strptr,
                                &data_not_used, a) != _RC_SUCCESS)
    return _RC_ERROR_INVALID_REQUEST;

  if (hostrpc_call_fnptr<T, FT>(NumArgs, fnptr, a, return_value) != _RC_SUCCESS)
    return _RC_ERROR_INVALID_REQUEST;

  return _RC_SUCCESS;
}
