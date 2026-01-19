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

#include "Emissary.h"
#include "EmissaryIds.h"
#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"
#include <unordered_map>

extern "C" EmissaryReturn_t
EmissaryTop(char *data, emisArgBuf_t *ab,
            std::unordered_map<void *, void *> *D2HAddrList) {
  EmissaryReturn_t result = 0;
  emis_argptr_t **args = (emis_argptr_t **)aligned_alloc(
      sizeof(emis_argptr_t), ab->NumArgs * sizeof(emis_argptr_t *));

  switch (ab->emisid) {
  case EMIS_ID_INVALID: {
    fprintf(stderr, "Emissary (host execution) got invalid EMIS_ID\n");
    result = 0;
    break;
  }
  case EMIS_ID_FORTRT: {
#ifdef EMISSARY_FLANGRT_SUPPORT
    if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                           &(ab->data_not_used), &args[0],
                           D2HAddrList) != _ERC_SUCCESS)
      return (EmissaryReturn_t)0;
    result = EmissaryFortrt(data, ab, args);
#else
    result = 0;
#endif
    break;
  }
  case EMIS_ID_PRINT: {
    result = EmissaryPrint(data, ab);
    break;
  }
  case EMIS_ID_MPI: {
    if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                           &(ab->data_not_used), &args[0],
                           D2HAddrList) != _ERC_SUCCESS)
      return (EmissaryReturn_t)0;
    result = EmissaryMPI(data, ab, args);
    break;
  }
  case EMIS_ID_HDF5: {
    if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                           &(ab->data_not_used), &args[0],
                           D2HAddrList) != _ERC_SUCCESS)
      return (EmissaryReturn_t)0;
    result = EmissaryHDF5(data, ab, args);
    break;
  }
  case EMIS_ID_RESERVE: {
    if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                           &(ab->data_not_used), &args[0],
                           D2HAddrList) != _ERC_SUCCESS)
      return (EmissaryReturn_t)0;
    result = EmissaryReserve(data, ab, args);
    break;
  }
  default:
    fprintf(stderr,
            "Emissary (host execution) EMIS_ID:%d fnid:%d not supported\n",
            ab->emisid, ab->emisfnid);
  }
  free(args);
  return result;
}

// emisExtractArgBuf reverses protocol that codegen in EmitEmissaryExec makes.
extern "C" void emisExtractArgBuf(char *data, emisArgBuf_t *ab) {

  uint32_t *int32_data = (uint32_t *)data;
  ab->DataLen = int32_data[0];
  ab->NumArgs = int32_data[1];

  // Note: while the data buffer contains all args including strings,
  // ab->DataLen does not include strings. It only counts header, keys,
  // and aligned numerics.

  ab->keyptr = data + (2 * sizeof(int));
  ab->argptr = ab->keyptr + (ab->NumArgs * sizeof(int));
  ab->strptr = data + (size_t)ab->DataLen;
  int alignfill = 0;
  if (((size_t)ab->argptr) % (size_t)8) {
    ab->argptr += 4;
    alignfill = 4;
  }

  // Extract the two emissary identifiers and number of send
  // and recv device data transfers. These are 4 16 bit values
  // packed into a single 64-bit field.
  uint64_t arg1 = *(uint64_t *)ab->argptr;
  ab->emisid = (unsigned int)((arg1 >> 48) & 0xFFFF);
  ab->emisfnid = (unsigned int)((arg1 >> 32) & 0xFFFF);
  ab->NumSendXfers = (unsigned int)((arg1 >> 16) & 0xFFFF);
  ab->NumRecvXfers = (unsigned int)((arg1) & 0xFFFF);

  // skip the uint64_t emissary id arg which is first arg in _emissary_exec.
  ab->keyptr += sizeof(int);
  ab->argptr += sizeof(uint64_t);
  ab->NumArgs -= 1;

  // data_not_used used for testing consistency.
  ab->data_not_used =
      (size_t)(ab->DataLen) - (((size_t)(3 + ab->NumArgs) * sizeof(int)) +
                               sizeof(uint64_t) + alignfill);

  // Ensure first arg after emissary id arg is aligned.
  if (((size_t)ab->argptr) % (size_t)8) {
    ab->argptr += 4;
    ab->data_not_used -= 4;
  }
}

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
extern "C" uint32_t
EmissaryBuildVargs(int NumArgs, char *keyptr, char *dataptr, char *strptr,
                   unsigned long long *data_not_used, emis_argptr_t *a[],
                   std::unordered_map<void *, void *> *D2HAddrList) {
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
        return _ERC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (emis_argptr_t *)getuint32(dataptr);
      else
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
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
        return _ERC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (emis_argptr_t *)getuint32(dataptr);
      else
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      break;

    case PointerTyID: {   ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t)*(unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        a[argcount] = (emis_argptr_t *)((char *)strptr);
      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      }
      if (D2HAddrList) {
        auto found = D2HAddrList->find((void *)a[argcount]);
        if (found != D2HAddrList->end())
          a[argcount] = (emis_argptr_t *)found->second;
      }
    } break;

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
      return _ERC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return _ERC_INVALID_ID_ERROR;
    }
    // Move to next argument
    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
    argcount++;
  }
  return _ERC_SUCCESS;
}

extern "C" void emisSkipXferArgSet(emisArgBuf_t *ab) {
  // Skip the ptr and size of the Xfer
  ab->NumArgs -= 2;
  ab->keyptr += 2 * sizeof(uint32_t);
  ab->argptr += 2 * sizeof(void *);
  ab->data_not_used -= 2 * sizeof(void *);
}
