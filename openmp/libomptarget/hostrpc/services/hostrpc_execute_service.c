
/*
 *   hostrpc_execute_service.c:  These are the host services for the hostrpc
system
 *
 *   Written by Greg Rodgers

MIT License

Copyright © 2020 Advanced Micro Devices, Inc.

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

#include "../src/hostrpc.h"
#include "hostrpc_internal.h"
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// MAXVARGS is more than a static array size.
// It is for user vargs functions only.
// It does not apply to printf.
#define MAXVARGS 32

// We would like to get llvm typeID enum from Type.h. e.g.
// #include "../../../../../llvm/include/llvm/IR/Type.h"
// But we cannot include LLVM headers in a runtime function.
// So for now, we a have a manual copy of llvm TypeID enum.

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
  IntegerTyID,       ///< Arbitrary bit width integers
  FunctionTyID,      ///< Functions
  PointerTyID,       ///< Pointers
  StructTyID,        ///< Structures
  ArrayTyID,         ///< Arrays
  FixedVectorTyID,   ///< Fixed width SIMD vector type
  ScalableVectorTyID ///< Scalable SIMD vector type
};

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
typedef uint32_t hostrpc_varfn_uint_t(void *, ...);
typedef uint64_t hostrpc_varfn_uint64_t(void *, ...);
typedef double hostrpc_varfn_double_t(void *, ...);

static hostrpc_status_t hostrpc_printf(char *buf, size_t bufsz, uint32_t *rc);
static hostrpc_status_t hostrpc_fprintf(char *buf, size_t bufsz, uint32_t *rc);
static hostrpc_status_t hostrpc_varfn_uint_(char *buf, size_t bufsz,
                                            uint32_t *rc);
static hostrpc_status_t hostrpc_varfn_uint64_(char *buf, size_t bufsz,
                                              uint64_t *rc);
static hostrpc_status_t hostrpc_varfn_double_(char *buf, size_t bufsz,
                                              double *rc);

static void hostrpc_handler_SERVICE_PRINTF(uint32_t device_id,
                                           uint64_t *payload) {
  size_t bufsz = (size_t)payload[0];
  char *device_buffer = (char *)payload[1];
  uint uint_value;
  hostrpc_status_t rc = hostrpc_printf(device_buffer, bufsz, &uint_value);
  payload[0] = (uint64_t)uint_value; // what the printf returns
  payload[1] = (uint64_t)rc;         // Any errors in the service function
  impl_free(device_buffer);
}
static void hostrpc_handler_SERVICE_FPRINTF(uint32_t device_id,
                                            uint64_t *payload) {
  size_t bufsz = (size_t)payload[0];
  char *device_buffer = (char *)payload[1];
  uint uint_value;
  hostrpc_status_t rc = hostrpc_fprintf(device_buffer, bufsz, &uint_value);
  payload[0] = (uint64_t)uint_value; // what the printf returns
  payload[1] = (uint64_t)rc;         // Any errors in the service function
  impl_free(device_buffer);
}

static void hostrpc_handler_SERVICE_VARFNUINT(uint32_t device_id,
                                              uint64_t *payload) {
  size_t bufsz = (size_t)payload[0];
  char *device_buffer = (char *)payload[1];
  uint uint_value;
  hostrpc_status_t rc = hostrpc_varfn_uint_(device_buffer, bufsz, &uint_value);
  payload[0] = (uint64_t)uint_value; // What the vargs function pointer returns
  payload[1] = (uint64_t)rc;         // any errors in the service function
  impl_free(device_buffer);
}

static void hostrpc_handler_SERVICE_VARFNUINT64(uint32_t device_id,
                                                uint64_t *payload) {
  size_t bufsz = (size_t)payload[0];
  char *device_buffer = (char *)payload[1];
  uint64_t uint64_value;
  hostrpc_status_t rc =
      hostrpc_varfn_uint64_(device_buffer, bufsz, &uint64_value);
  payload[0] =
      (uint64_t)uint64_value; // What the vargs function pointer returns
  payload[1] = (uint64_t)rc;  // any errors in the service function
  impl_free(device_buffer);
}

static void hostrpc_handler_SERVICE_VARFNDOUBLE(uint32_t device_id,
                                                uint64_t *payload) {
  size_t bufsz = (size_t)payload[0];
  char *device_buffer = (char *)payload[1];
  double double_value;
  hostrpc_status_t rc =
      hostrpc_varfn_double_(device_buffer, bufsz, (double *)&double_value);
  memcpy(&payload[0], &double_value, 8);
  payload[1] = (uint64_t)rc; // any errors in the service function
  impl_free(device_buffer);
}

static void hostrpc_handler_SERVICE_MALLOC_PRINTF(uint32_t device_id,
                                                  uint64_t *payload) {
  void *ptr = NULL;
  // CPU device ID 0 is the fine grain memory
  size_t sz = (size_t)payload[0];
  hsa_status_t err = host_malloc(&ptr, sz);
  payload[0] = (uint64_t)err;
  payload[1] = (uint64_t)ptr;
}

static void hostrpc_handler_SERVICE_MALLOC(uint32_t device_id,
                                           uint64_t *payload) {
  void *ptr = NULL;
  hsa_status_t err = device_malloc(&ptr, payload[0], device_id);
  payload[0] = (uint64_t)err;
  payload[1] = (uint64_t)ptr;
}

static void hostrpc_handler_SERVICE_FTNASSIGN(uint32_t device_id,
                                              uint64_t *payload) {
  void *ptr = NULL;
  hsa_status_t err = ftn_assign_wrapper(payload[0], payload[1], payload[2], payload[3], payload[4]);
  payload[0] = (uint64_t)err;
  payload[1] = (uint64_t)ptr;
}


static void hostrpc_handler_SERVICE_FREE(uint32_t device_id,
                                         uint64_t *payload) {
  char *device_buffer = (char *)payload[0];
  impl_free(device_buffer);
}

static void hostrpc_handler_SERVICE_FUNCTIONCALL(uint32_t device_id,
                                                 uint64_t *payload) {
  void (*fptr)() = (void *)payload[0];
  (*fptr)();
}

// This is the host function for the demo vector_product_zeros
static int local_vector_product_zeros(int N, int *A, int *B, int *C) {
  int zeros = 0;
  for (int i = 0; i < N; i++) {
    C[i] = A[i] * B[i];
    if (C[i] == 0)
      zeros++;
  }
  return zeros;
}

// This is the service for the demo of vector_product_zeros
static void hostrpc_handler_SERVICE_DEMO(uint32_t device_id,
                                         uint64_t *payload) {
  hsa_status_t copyerr;
  int N = (int)payload[0];
  int *A_D = (int *)payload[1];
  int *B_D = (int *)payload[2];
  int *C_D = (int *)payload[3];

  int *A = (int *)malloc(N * sizeof(int));
  int *B = (int *)malloc(N * sizeof(int));
  int *C = (int *)malloc(N * sizeof(int));
  copyerr = impl_memcpy_no_signal(A, A_D, N * sizeof(int), false);
  copyerr = impl_memcpy_no_signal(B, B_D, N * sizeof(int), false);

  int num_zeros = local_vector_product_zeros(N, A, B, C);
  copyerr = impl_memcpy_no_signal(C_D, C, N * sizeof(int), true);
  payload[0] = (uint64_t)copyerr;
  payload[1] = (uint64_t)num_zeros;
}

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

static void hostrpc_abort(int rc) {
  printf("hostrpc_abort called with code %d\n", rc);
  abort();
}

// The architecture-specific implementation of hostrpc will
// call this single external function for each service request.
// Host service functions are architecturally independent.
extern void hostrpc_execute_service(uint32_t service, uint32_t device_id,
                                    uint64_t *payload) {

  // split the 32-bit service number into service_id and VRM to be checked
  // if device hostrpc or stubs are ahead of this host runtime.
  uint service_id = (service << 16) >> 16;
  if (!hostrpc_version_checked) {
    uint device_vrm = ((uint)service >> 16);
    hostrpc_status_t err = hostrpc_version_check(device_vrm);
    if (err != HOSTRPC_SUCCESS)
      hostrpc_abort(err);
    hostrpc_version_checked = true;
  }

  switch (service_id) {
  case HOSTRPC_SERVICE_PRINTF:
    hostrpc_handler_SERVICE_PRINTF(device_id, payload);
    break;
  case HOSTRPC_SERVICE_FPRINTF:
    hostrpc_handler_SERVICE_FPRINTF(device_id, payload);
    break;
  case HOSTRPC_SERVICE_VARFNUINT:
    hostrpc_handler_SERVICE_VARFNUINT(device_id, payload);
    break;
  case HOSTRPC_SERVICE_VARFNUINT64:
    hostrpc_handler_SERVICE_VARFNUINT64(device_id, payload);
    break;
  case HOSTRPC_SERVICE_VARFNDOUBLE:
    hostrpc_handler_SERVICE_VARFNDOUBLE(device_id, payload);
    break;
  case HOSTRPC_SERVICE_MALLOC_PRINTF:
    hostrpc_handler_SERVICE_MALLOC_PRINTF(device_id, payload);
    break;
  case HOSTRPC_SERVICE_MALLOC:
    hostrpc_handler_SERVICE_MALLOC(device_id, payload);
    break;
  case HOSTRPC_SERVICE_FTNASSIGN:
    hostrpc_handler_SERVICE_FTNASSIGN(device_id, payload);
    break;
  case HOSTRPC_SERVICE_FREE:
    hostrpc_handler_SERVICE_FREE(device_id, payload);
    break;
  case HOSTRPC_SERVICE_FUNCTIONCALL:
    hostrpc_handler_SERVICE_FUNCTIONCALL(device_id, payload);
    break;
  case HOSTRPC_SERVICE_DEMO:
    hostrpc_handler_SERVICE_DEMO(device_id, payload);
    break;
  default:
    hostrpc_abort(HOSTRPC_INVALIDSERVICE_ERROR);
    printf("ERROR: hostrpc got a bad service id:%d\n", service);
  }
}

//---------------- Support for hostrpc_printf service ---------------------

// Handle overflow when building the va_list for vprintf
static hostrpc_status_t hostrpc_pfGetOverflow(hostrpc_ValistExt_t *valist,
                                              size_t needsize) {
  if (needsize < valist->overflow_size)
    return HOSTRPC_SUCCESS;

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
    return HOSTRPC_STATUS_ERROR;
  }
  memset(newstack, 0, stacksize);
  if (valist->overflow_size) {
    memcpy(newstack, valist->overflow_arg_area, valist->overflow_size);
    free(valist->overflow_arg_area);
  }
  valist->overflow_arg_area = newstack;
  valist->overflow_size = stacksize;
  return HOSTRPC_SUCCESS;
}

// Add an integer to the va_list for vprintf
static hostrpc_status_t hostrpc_pfAddInteger(hostrpc_ValistExt_t *valist,
                                             char *val, size_t valsize,
                                             size_t *stacksize) {
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
    return HOSTRPC_STATUS_ERROR;
  }
  }
  //  Always copy 8 bytes, sizeof(ival)
  if ((valist->gp_offset + sizeof(ival)) <= sizeof(hostrpc_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), &ival,
           sizeof(ival));
    valist->gp_offset += sizeof(ival);
    return HOSTRPC_SUCCESS;
  }
  // Ensure valist overflow area is big enough
  size_t needsize = (size_t)*stacksize + sizeof(ival);
  if (hostrpc_pfGetOverflow(valist, needsize) != HOSTRPC_SUCCESS)
    return HOSTRPC_STATUS_ERROR;
  // Copy to overflow
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &ival,
         sizeof(ival));

  *stacksize += sizeof(ival);
  return HOSTRPC_SUCCESS;
}

// Add a String argument when building va_list for vprintf
static hostrpc_status_t hostrpc_pfAddString(hostrpc_ValistExt_t *valist,
                                            char *val, size_t strsz,
                                            size_t *stacksize) {
  size_t valsize =
      sizeof(char *); // ABI captures pointer to string,  not string
  if ((valist->gp_offset + valsize) <= sizeof(hostrpc_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), val, valsize);
    valist->gp_offset += valsize;
    return HOSTRPC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + valsize;
  if (hostrpc_pfGetOverflow(valist, needsize) != HOSTRPC_SUCCESS)
    return HOSTRPC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, val,
         valsize);
  *stacksize += valsize;
  return HOSTRPC_SUCCESS;
}

// Add a floating point value when building va_list for vprintf
static hostrpc_status_t hostrpc_pfAddFloat(hostrpc_ValistExt_t *valist,
                                           char *numdata, size_t valsize,
                                           size_t *stacksize) {
  // FIXME, we can used load because doubles are now aligned
  double dval;
  if (valsize == 4) {
    float fval;
    memcpy(&fval, numdata, 4);
    dval = (double)fval; // Extend single to double per abi
  } else if (valsize == 8) {
    memcpy(&dval, numdata, 8);
  } else {
    return HOSTRPC_STATUS_ERROR;
  }
  if ((valist->fp_offset + FPREGSZ) <= sizeof(hostrpc_pfRegSaveArea_t)) {
    memcpy(((char *)valist->reg_save_area + (size_t)(valist->fp_offset)), &dval,
           sizeof(double));
    valist->fp_offset += FPREGSZ;
    return HOSTRPC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + sizeof(double);
  if (hostrpc_pfGetOverflow(valist, needsize) != HOSTRPC_SUCCESS)
    return HOSTRPC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &dval,
         sizeof(double));
  // move only by the size of the double (8 bytes)
  *stacksize += sizeof(double);
  return HOSTRPC_SUCCESS;
}

// Build an extended va_list for vprintf by unpacking the buffer
static hostrpc_status_t hostrpc_pfBuildValist(hostrpc_ValistExt_t *valist,
                                              int NumArgs, char *keyptr,
                                              char *dataptr, char *strptr,
                                              size_t *data_not_used) {
  hostrpc_pfRegSaveArea_t *regs;
  size_t regs_size = sizeof(*regs);
  regs = (hostrpc_pfRegSaveArea_t *)malloc(regs_size);
  if (!regs)
    return HOSTRPC_STATUS_ERROR;
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
        return HOSTRPC_DATA_USED_ERROR;
      if (valist->fp_offset == 0)
        valist->fp_offset = sizeof(hostrpc_pfIntRegs_t);
      if (hostrpc_pfAddFloat(valist, dataptr, num_bytes, &stacksize))
        return HOSTRPC_ADDFLOAT_ERROR;
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
        return HOSTRPC_DATA_USED_ERROR;
      if (hostrpc_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
        return HOSTRPC_ADDINT_ERROR;
      break;

    case PointerTyID:     ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t) * (unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return HOSTRPC_DATA_USED_ERROR;
        if (hostrpc_pfAddString(valist, (char *)&strptr, strsz, &stacksize))
          return HOSTRPC_ADDSTRING_ERROR;
      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return HOSTRPC_DATA_USED_ERROR;
        if (hostrpc_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
          return HOSTRPC_ADDINT_ERROR;
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
    case VoidTyID:
      return HOSTRPC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return HOSTRPC_INVALID_ID_ERROR;
    }

    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
  }
  return HOSTRPC_SUCCESS;
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
static hostrpc_status_t hostrpc_fprintf(char *buf, size_t bufsz, uint *rc) {

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
                            &data_not_used) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer for hostrpc_varfn_uint to consume
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(hostrpc_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;

  *rc = vfprintf(fileptr, fmtstr, *real_va_list);

  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);

  return HOSTRPC_SUCCESS;
}
//  This the main service routine for printf
static hostrpc_status_t hostrpc_printf(char *buf, size_t bufsz, uint *rc) {
  if (bufsz == 0)
    return HOSTRPC_SUCCESS;

  // Get 6 values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  if (NumArgs <= 0)
    return HOSTRPC_ERROR_INVALID_REQUEST;

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
                            &data_not_used) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer for hostrpc_varfn_uint to consume
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(hostrpc_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;

  *rc = vprintf(fmtstr, *real_va_list);

  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);

  return HOSTRPC_SUCCESS;
}

//---------------- Support for hostrpc_varfn_* service ---------------------
//

// These are the helper functions for hostrpc_varfn_uint_
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
static hostrpc_status_t hostrpc_build_vargs_array(int NumArgs, char *keyptr,
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
        return HOSTRPC_DATA_USED_ERROR;

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
        return HOSTRPC_DATA_USED_ERROR;

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
          return HOSTRPC_DATA_USED_ERROR;
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
          return HOSTRPC_DATA_USED_ERROR;

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
    case VoidTyID:
      return HOSTRPC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return HOSTRPC_INVALID_ID_ERROR;
    }

    // Move to next argument
    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
    argcount++;
  }
  return HOSTRPC_SUCCESS;
}

// Make the vargs function call to the function pointer fnptr
// by casting fnptr to vfnptr. Return uint32_t
static hostrpc_status_t hostrpc_call_fnptr_uint(uint32_t NumArgs, void *fnptr,
                                                uint64_t *a[MAXVARGS],
                                                uint32_t *rc) {
  //
  // Users are instructed that their first arg must be a dummy
  // so that device interface is same as host interface. To match device
  // interface we make the first arg be the function pointer.
  //
  hostrpc_varfn_uint_t *vfnptr = (hostrpc_varfn_uint_t *)fnptr;

  switch (NumArgs) {
  case 1:
    *rc = vfnptr(fnptr, a[0]);
    break;
  case 2:
    *rc = vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    *rc = vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9]);
    break;
  case 11:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10]);
    break;
  case 12:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11]);
    break;
  case 13:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12]);
    break;
  case 14:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18]);
    break;
  case 20:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19]);
    break;
  case 21:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20]);
    break;
  case 22:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21]);
    break;
  case 23:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
    break;
  case 28:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27]);
    break;
  case 29:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28]);
    break;
  case 30:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29]);
    break;
  case 31:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30]);
    break;
  case 32:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    return HOSTRPC_EXCEED_MAXVARGS_ERROR;
  }
  return HOSTRPC_SUCCESS;
}

// Make the vargs function call to the function pointer fnptr
// by casting fnptr to vfnptr. Return uint64
static hostrpc_status_t hostrpc_call_fnptr_uint64(uint32_t NumArgs, void *fnptr,
                                                  uint64_t *a[MAXVARGS],
                                                  uint64_t *rc) {
  //
  // Users are instructed that their first arg must be a dummy
  // so that device interface is same as host interface. To match device
  // interface we make the first arg be the function pointer.
  //
  hostrpc_varfn_uint64_t *vfnptr = (hostrpc_varfn_uint64_t *)fnptr;

  switch (NumArgs) {
  case 1:
    *rc = vfnptr(fnptr, a[0]);
    break;
  case 2:
    *rc = vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    *rc = vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9]);
    break;
  case 11:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10]);
    break;
  case 12:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11]);
    break;
  case 13:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12]);
    break;
  case 14:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18]);
    break;
  case 20:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19]);
    break;
  case 21:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20]);
    break;
  case 22:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21]);
    break;
  case 23:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
    break;
  case 28:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27]);
    break;
  case 29:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28]);
    break;
  case 30:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29]);
    break;
  case 31:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30]);
    break;
  case 32:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    return HOSTRPC_EXCEED_MAXVARGS_ERROR;
  }
  return HOSTRPC_SUCCESS;
}

// Make the vargs function call to the function pointer fnptr
// by casting fnptr to vfnptr. Return double
static hostrpc_status_t hostrpc_call_fnptr_double(uint32_t NumArgs, void *fnptr,
                                                  uint64_t *a[MAXVARGS],
                                                  double *rc) {
  //
  // Users are instructed that their first arg must be a dummy
  // so that device interface is same as host interface. To match device
  // interface we make the first arg be the function pointer.
  //
  hostrpc_varfn_double_t *vfnptr = (hostrpc_varfn_double_t *)fnptr;

  switch (NumArgs) {
  case 1:
    *rc = vfnptr(fnptr, a[0]);
    break;
  case 2:
    *rc = vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    *rc = vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9]);
    break;
  case 11:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10]);
    break;
  case 12:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11]);
    break;
  case 13:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12]);
    break;
  case 14:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18]);
    break;
  case 20:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19]);
    break;
  case 21:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20]);
    break;
  case 22:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21]);
    break;
  case 23:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26]);
    break;
  case 28:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27]);
    break;
  case 29:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28]);
    break;
  case 30:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29]);
    break;
  case 31:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30]);
    break;
  case 32:
    *rc = vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                 a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                 a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25], a[26],
                 a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    return HOSTRPC_EXCEED_MAXVARGS_ERROR;
  }
  return HOSTRPC_SUCCESS;
}
//  This the main service routine for hostrpc_varfn_uint
static hostrpc_status_t hostrpc_varfn_uint_(char *buf, size_t bufsz, uint *rc) {
  if (bufsz == 0)
    return HOSTRPC_SUCCESS;

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
    return HOSTRPC_ERROR_INVALID_REQUEST;

  uint64_t *a[MAXVARGS];
  if (hostrpc_build_vargs_array(NumArgs, keyptr, dataptr, strptr,
                                &data_not_used, a) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  if (hostrpc_call_fnptr_uint(NumArgs, fnptr, a, rc) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  return HOSTRPC_SUCCESS;
}

//  This the main service routine for hostrpc_varfn_uint64
static hostrpc_status_t hostrpc_varfn_uint64_(char *buf, size_t bufsz,
                                              uint64_t *rc) {
  if (bufsz == 0)
    return HOSTRPC_SUCCESS;

  // Get 6 tracking values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  if (NumArgs <= 0)
    return HOSTRPC_ERROR_INVALID_REQUEST;

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

  uint64_t *a[MAXVARGS];
  if (hostrpc_build_vargs_array(NumArgs, keyptr, dataptr, strptr,
                                &data_not_used, a) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  if (hostrpc_call_fnptr_uint64(NumArgs, fnptr, a, rc) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  return HOSTRPC_SUCCESS;
}

//  This the main service routine for hostrpc_varfn_double
static hostrpc_status_t hostrpc_varfn_double_(char *buf, size_t bufsz,
                                              double *rc) {
  if (bufsz == 0)
    return HOSTRPC_SUCCESS;

  // Get 6 tracking values needed to unpack the buffer
  int *datalen = (int *)buf;
  int NumArgs = *((int *)(buf + sizeof(int)));
  size_t data_not_used =
      (size_t)(*datalen) - ((size_t)(2 + NumArgs) * sizeof(int));
  char *keyptr = buf + (2 * sizeof(int));
  char *dataptr = keyptr + (NumArgs * sizeof(int));
  char *strptr = buf + (size_t)*datalen;

  if (NumArgs <= 0)
    return HOSTRPC_ERROR_INVALID_REQUEST;

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

  uint64_t *a[MAXVARGS];
  if (hostrpc_build_vargs_array(NumArgs, keyptr, dataptr, strptr,
                                &data_not_used, a) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  if (hostrpc_call_fnptr_double(NumArgs, fnptr, a, rc) != HOSTRPC_SUCCESS)
    return HOSTRPC_ERROR_INVALID_REQUEST;

  return HOSTRPC_SUCCESS;
}
