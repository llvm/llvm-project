//===--- offload/plugins-nextgen/common/src/EmissaryPrint.cpp ----- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Host support for misc emissary API. 
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <cstring>
#include <ctype.h>
#include <list>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../../DeviceRTL/include/EmissaryIds.h"
#include "Emissary.h"

static service_rc emissary_printf(uint *rc, emisArgBuf_t *ab);
static service_rc emissary_fprintf(uint *rc, emisArgBuf_t *ab);

extern "C" emis_return_t EmissaryPrint(char *data, emisArgBuf_t *ab) {
  uint32_t return_value;
  service_rc rc;
  switch (ab->emisfnid) {
  case _printf_idx: {
    rc = emissary_printf(&return_value, ab);
    break;
  }
  case _fprintf_idx: {
    rc = emissary_fprintf(&return_value, ab);
    break;
  }
  case _ockl_asan_report_idx: {
    fprintf(stderr, " asan_report not yet implemented\n");
    return_value = 0;
    rc = _RC_STATUS_ERROR;
    break;
  }
  case _print_INVALID:
  default: {
    fprintf(stderr, " INVALID emissary function id (%d) for PRINT API \n",
            ab->emisfnid);
    return_value = 0;
    rc = _RC_STATUS_ERROR;
    break;
  }
  }
  if (rc != _RC_SUCCESS)
    fprintf(stderr, "HOST failure in _emissary_execute_print\n");

  return (emis_return_t)return_value;
}

// NUMFPREGS and FPREGSZ are part of x86 vargs ABI that
// is recreated with the printf support.
#define NUMFPREGS 8
#define FPREGSZ 16

typedef int uint128_t __attribute__((mode(TI)));
struct emissary_pfIntRegs {
  uint64_t rdi, rsi, rdx, rcx, r8, r9;
};
typedef struct emissary_pfIntRegs emissary_pfIntRegs_t; // size = 48 bytes

struct emissary_pfRegSaveArea {
  emissary_pfIntRegs_t iregs;
  uint128_t freg[NUMFPREGS];
};
typedef struct emissary_pfRegSaveArea
    emissary_pfRegSaveArea_t; // size = 304 bytes

struct emissary_ValistExt {
  uint32_t gp_offset;      /* offset to next available gpr in reg_save_area */
  uint32_t fp_offset;      /* offset to next available fpr in reg_save_area */
  void *overflow_arg_area; /* args that are passed on the stack */
  emissary_pfRegSaveArea_t *reg_save_area; /* int and fp registers */
  size_t overflow_size;
} __attribute__((packed));
typedef struct emissary_ValistExt emissary_ValistExt_t;

// Handle overflow when building the va_list for vprintf
static service_rc emissary_pfGetOverflow(emissary_ValistExt_t *valist,
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
static service_rc emissary_pfAddInteger(emissary_ValistExt_t *valist, char *val,
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
  if ((valist->gp_offset + sizeof(ival)) <= sizeof(emissary_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), &ival,
           sizeof(ival));
    valist->gp_offset += sizeof(ival);
    return _RC_SUCCESS;
  }
  // Ensure valist overflow area is big enough
  size_t needsize = (size_t)*stacksize + sizeof(ival);
  if (emissary_pfGetOverflow(valist, needsize) != _RC_SUCCESS)
    return _RC_STATUS_ERROR;
  // Copy to overflow
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &ival,
         sizeof(ival));

  *stacksize += sizeof(ival);
  return _RC_SUCCESS;
}

// Add a String argument when building va_list for vprintf
static service_rc emissary_pfAddString(emissary_ValistExt_t *valist, char *val,
                                       size_t strsz, size_t *stacksize) {
  size_t valsize =
      sizeof(char *); // ABI captures pointer to string,  not string
  if ((valist->gp_offset + valsize) <= sizeof(emissary_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), val, valsize);
    valist->gp_offset += valsize;
    return _RC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + valsize;
  if (emissary_pfGetOverflow(valist, needsize) != _RC_SUCCESS)
    return _RC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, val,
         valsize);
  *stacksize += valsize;
  return _RC_SUCCESS;
}

// Add a floating point value when building va_list for vprintf
static service_rc emissary_pfAddFloat(emissary_ValistExt_t *valist,
                                      char *numdata, size_t valsize,
                                      size_t *stacksize) {
  // we could use load because doubles are now aligned
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
  if ((valist->fp_offset + FPREGSZ) <= sizeof(emissary_pfRegSaveArea_t)) {
    memcpy(((char *)valist->reg_save_area + (size_t)(valist->fp_offset)), &dval,
           sizeof(double));
    valist->fp_offset += FPREGSZ;
    return _RC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + sizeof(double);
  if (emissary_pfGetOverflow(valist, needsize) != _RC_SUCCESS)
    return _RC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &dval,
         sizeof(double));
  // move only by the size of the double (8 bytes)
  *stacksize += sizeof(double);
  return _RC_SUCCESS;
}

// Build an extended va_list for vprintf by unpacking the buffer
static service_rc emissary_pfBuildValist(emissary_ValistExt_t *valist,
                                         int NumArgs, char *keyptr,
                                         char *dataptr, char *strptr,
                                         size_t *data_not_used) {
  emissary_pfRegSaveArea_t *regs;
  size_t regs_size = sizeof(*regs);
  regs = (emissary_pfRegSaveArea_t *)malloc(regs_size);
  if (!regs)
    return _RC_STATUS_ERROR;
  memset(regs, 0, regs_size);
  *valist = (emissary_ValistExt_t){
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
        valist->fp_offset = sizeof(emissary_pfIntRegs_t);
      if (emissary_pfAddFloat(valist, dataptr, num_bytes, &stacksize))
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
      if (emissary_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
        return _RC_ADDINT_ERROR;
      break;

    case PointerTyID:     ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t)*(unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _RC_DATA_USED_ERROR;
        if (emissary_pfAddString(valist, (char *)&strptr, strsz, &stacksize))
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
        if (emissary_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
          return _RC_ADDINT_ERROR;
      }
      break;

    case HalfTyID:           ///<  1: 16-bit floating point type
    case ArrayTyID:          ///< 14: Arrays
    case StructTyID:         ///< 13: Structures
    case FunctionTyID:       ///< 12: Functions
    case TokenTyID:          ///< 10: Tokens
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
} // end emissary_pfBuildValist

/*
 *  The buffer to pack arguments for all vargs functions has thes 4 sections:
 *  1. Header  datalen 4 bytes
 *             numargs 4 bytes
 *  2. keyptr  A 4-byte key for each arg including string args
 *             Each 4-byte key contains llvmID and numbits to
 *             describe the datatype.
 *  3. argptr  Ths data values for each argument.
 *             Each arg is aligned according to its size.
 *             If the field is a string
 *             the dataptr contains the string length.
 *  4. strptr  Exection time string values
 */
static service_rc emissary_fprintf(uint *rc, emisArgBuf_t *ab) {

  if (ab->DataLen == 0)
    return _RC_SUCCESS;

  char *fmtstr = ab->strptr;
  FILE *fileptr = (FILE *)*((size_t *)ab->argptr);

  // Skip past the file pointer
  ab->NumArgs--;
  ab->keyptr += 4;
  ab->argptr += sizeof(FILE *);
  ab->data_not_used -= sizeof(FILE *);

  // Skip past the format string
  ab->NumArgs--;
  ab->keyptr += 4;
  size_t abstrsz = (size_t)*(unsigned int *)ab->argptr;
  ab->strptr += abstrsz;
  ab->argptr += 4;
  ab->data_not_used -= 4;

  emissary_ValistExt_t valist;
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (emissary_pfBuildValist(&valist, ab->NumArgs, ab->keyptr, ab->argptr,
                             ab->strptr, &ab->data_not_used) != _RC_SUCCESS)
    return _RC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(emissary_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;

  *rc = vfprintf(fileptr, fmtstr, *real_va_list);

  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);

  return _RC_SUCCESS;
}

static service_rc emissary_printf(uint *rc, emisArgBuf_t *ab) {
  if (ab->DataLen == 0)
    return _RC_SUCCESS;

  char *fmtstr = ab->strptr;

  // Skip past the format string
  ab->NumArgs--;
  ab->keyptr += 4;
  size_t abstrsz = (size_t)*(unsigned int *)ab->argptr;
  ab->strptr += abstrsz;
  ab->argptr += 4;
  ab->data_not_used -= 4;

  emissary_ValistExt_t valist;
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (emissary_pfBuildValist(&valist, ab->NumArgs, ab->keyptr, ab->argptr,
                             ab->strptr, &ab->data_not_used) != _RC_SUCCESS)
    return _RC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer for
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(emissary_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;

  *rc = vprintf(fmtstr, *real_va_list);

  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);

  return _RC_SUCCESS;
}

extern "C" void *global_allocate(uint32_t bufsz) {
  return malloc((size_t)bufsz);
}
extern "C" int global_free(void *ptr) {
  free(ptr);
  return 0;
}
