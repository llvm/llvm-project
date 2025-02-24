//===----- ioffload/plugins-nexgen/common/include/Emissary.cpp ---- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RPC.h"

#include "Shared/Debug.h"
#include "Shared/RPCOpcodes.h"

#include "PluginInterface.h"

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"

#include "../../../DeviceRTL/include/EmissaryIds.h"
#include "Emissary.h"

extern "C" emis_return_t _emissary_execute(void *data) {

  uint32_t *int32_data = (uint32_t *)data;
  uint32_t sz = int32_data[0];
  // Note: while the data buffer contains all args including strings,
  // sz does not include strings. It only counts header, keys,
  // and aligned numerics.
  uint32_t nargs = int32_data[1];

  // Extract the two emissary identifiers from 1st 64bit arg
  char *char_data = (char *)data;
  char *argptr = char_data + ((nargs + 2) * sizeof(int));
  if (((size_t)argptr) % (size_t)8)
    argptr += 4;
  uint64_t emis_ids = *(uint64_t *)argptr;
  offload_emis_id_t emis_id = (offload_emis_id_t)((uint)(emis_ids >> 32));
  uint32_t emis_func_id = (uint32_t)((uint)((emis_ids << 32) >> 32));
  unsigned long long result;
  switch (emis_id) {
  case EMIS_ID_INVALID: {
    fprintf(stderr, "_emissary_execute got invalid EMIS_ID\n");
    result = 0;
    break;
  }
  case EMIS_ID_FORTRT: {
    result = _emissary_execute_fortrt(emis_func_id, data, sz);
    break;
  }
  case EMIS_ID_PRINT: {
    result = _emissary_execute_print(emis_func_id, data, sz);
    break;
  }
  case EMIS_ID_MPI: {
    // res = _emissary_execute_mpi(emis_func_id, data, sz);
    result = 0;
    fprintf(stderr, "Support for MPI Emissary API is in development\n");
    break;
  }
  case EMIS_ID_HDF5: {
    // result = _emissary_execute_hdf5(emis_func_id, data, sz);
    result = 0;
    fprintf(stderr, "Support for HDF5 Emissary API is in development\n");
    break;
  }
  default:
    fprintf(stderr, "EMIS_ID %d not supported, func_id=%d\n", emis_id,
            emis_func_id);
  }
  return result;
}

/// These are helper functions for host support of emissary APIs

/// Get uint32 value extended to uint64_t value from a char ptr
extern "C" uint64_t getuint32(char *val) {
  uint32_t i32 = *(uint32_t *)val;
  return (uint64_t)i32;
}

/// Get uint64_t value from a char ptr
extern "C" uint64_t getuint64(char *val) { return *(uint64_t *)val; }

/// Get a function pointer from a char ptr
extern "C" void *getfnptr(char *val) {
  uint64_t ival = *(uint64_t *)val;
  return (void *)ival;
}

// build argument array
extern "C" uint32_t _build_vargs_array(int NumArgs, char *keyptr, char *dataptr,
                                       char *strptr, size_t *data_not_used,
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
        strsz = (size_t)*(unsigned int *)dataptr;
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

// Host defines for f90print functions needed just for linking
// and fallback when used in a target region
extern "C" void f90print_(char *s) { printf("%s\n", s); }
extern "C" void f90printi_(char *s, int *i) { printf("%s %d\n", s, *i); }
extern "C" void f90printl_(char *s, long *i) { printf("%s %ld\n", s, *i); }
extern "C" void f90printf_(char *s, float *f) { printf("%s %f\n", s, *f); }
extern "C" void f90printd_(char *s, double *d) { printf("%s %g\n", s, *d); }

extern "C" void *rpc_allocate(uint64_t sz) {
  printf("HOST rpc_allocate\n");
  return nullptr;
}
extern "C" void rpc_free(void *ptr) {
  printf("HOST rpc_free\n");
  return;
}
