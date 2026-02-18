//===-- offload/plugins-nextgen/common/include/Emissary.h ------ C++ ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines emissary helper functions. This include is only used for host
// compilation.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_EMISSARY_H
#define OFFLOAD_EMISSARY_H

#include "EmissaryIds.h"
#include "RPC.h"
// #include "shared/rpc.h"
#include "shared/rpc_server.h"

extern "C" {

/// Called by rpc after receiving emissary argument buffer
EmissaryReturn_t EmissaryTop(char *data, emisArgBuf_t *ab,
                             std::unordered_map<void *, void *> *D2HAddrList);

#ifdef EMISSARY_FLANGRT_SUPPORT
/// Called by Emissary for all Fortrt emissary functions
EmissaryReturn_t EmissaryFortrt(char *data, emisArgBuf_t *ab,
                                emis_argptr_t *arg[]);
#endif

/// Called by Emissary for all misc print functions
EmissaryReturn_t EmissaryPrint(char *data, emisArgBuf_t *ab);

/// Called by Emissary for all MPI emissary API functions
__attribute((weak)) EmissaryReturn_t EmissaryMPI(char *data, emisArgBuf_t *ab,
                                                 emis_argptr_t *arg[]);

/// Called by Emissary for all HDF5 Emissary API functions
__attribute((weak)) EmissaryReturn_t EmissaryHDF5(char *data, emisArgBuf_t *ab,
                                                  emis_argptr_t *arg[]);

/// Called by Emissary to support user-defined emissary API
__attribute((weak)) EmissaryReturn_t EmissaryReserve(char *data,
                                                     emisArgBuf_t *ab,
                                                     emis_argptr_t *arg[]);

/// emisExtractArgBuf is called within the "case OFFLOAD_EMISSARY:" stanza
/// in offload/plugins-nextgen/common/src/RPC.cpp to build the emisArgBuf_t
/// structure from the emissary data buffer sent to the CPU by rpc.
/// This buffer is created by clang CodeGen when variadic function
/// _emissary_exec(...) is encountered when compiling any emissary device
/// stub to define the device function.
void emisExtractArgBuf(char *buf, emisArgBuf_t *ab);

/// Move the ArgBuf tracker past a set of XferSrgs
void emisSkipXferArgSet(emisArgBuf_t *ab);

/// Get uint32 value extended to uint64_t value from a char ptr
uint64_t getuint32(char *val);
/// Get uint64_t value from a char ptr
uint64_t getuint64(char *val);
/// Get a function pointer from a char ptr
void *getfnptr(char *val);

/// Builds the array of pointers passed to V_ functions
uint32_t EmissaryBuildVargs(int NumArgs, char *keyptr, char *dataptr,
                            char *strptr, unsigned long long *data_not_used,
                            emis_argptr_t *a[],
                            std::unordered_map<void *, void *> *D2HAddrList);

} // end extern "C"

/// Call the associated V_ function
template <typename T, typename FT>
extern T EmissaryCallFnptr(uint32_t NumArgs, void *fnptr, emis_argptr_t *a[]);

// We would like to get llvm typeID enum from Type.h. e.g.
// #include ".../llvm/include/llvm/IR/Type.h"
// But we cannot include LLVM headers in a runtime function.
// So we a have a manual copy of llvm TypeID enum from Type.h
// The codegen for _emissary_exec puts this ID in the key for
// each arg and the host runtime needs to decode this key.
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

template <typename T, typename FT>
extern T EmissaryCallFnptr(uint32_t NumArgs, void *fnptr, emis_argptr_t *a[]) {
  T rv;
  FT *vfnptr = (FT *)fnptr;
  switch (NumArgs) {
  case 1:
    rv = (T)vfnptr(fnptr, a[0]);
    break;
  case 2:
    rv = (T)vfnptr(fnptr, a[0], a[1]);
    break;
  case 3:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2]);
    break;
  case 4:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3]);
    break;
  case 5:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4]);
    break;
  case 6:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    break;
  case 7:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    break;
  case 8:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    break;
  case 9:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
    break;
  case 10:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9]);
    break;
  case 11:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10]);
    break;
  case 12:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11]);
    break;
  case 13:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12]);
    break;
  case 14:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13]);
    break;
  case 15:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14]);
    break;
  case 16:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15]);
    break;
  case 17:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16]);
    break;
  case 18:
    rv =
        (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                  a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17]);
    break;
  case 19:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18]);
    break;
  case 20:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19]);
    break;
  case 21:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20]);
    break;
  case 22:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21]);
    break;
  case 23:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22]);
    break;
  case 24:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23]);
    break;
  case 25:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23], a[24]);
    break;
  case 26:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25]);
    break;
  case 27:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25],
                   a[26]);
    break;
  case 28:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25],
                   a[26], a[27]);
    break;
  case 29:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25],
                   a[26], a[27], a[28]);
    break;
  case 30:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25],
                   a[26], a[27], a[28], a[29]);
    break;
  case 31:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25],
                   a[26], a[27], a[28], a[29], a[30]);
    break;
  case 32:
    rv = (T)vfnptr(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                   a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17],
                   a[18], a[19], a[20], a[21], a[22], a[23], a[24], a[25],
                   a[26], a[27], a[28], a[29], a[30], a[31]);
    break;
  default:
    rv = 0;
  }
  return rv;
}

#endif // OFFLOAD_EMISSARY_H
