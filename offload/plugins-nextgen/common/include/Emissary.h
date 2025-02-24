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

typedef unsigned long long emis_return_t;
typedef uint64_t emis_uint64_t(void *, ...);
typedef uint32_t emis_uint32_t(void *, ...);

// MAXVARGS is the maximum number of args in an emissary function
// To increase this number be sure to updated _call_fnptr
#define MAXVARGS 32

extern "C" {
emis_return_t _emissary_execute(void *data);
emis_return_t _emissary_execute_fortrt(uint32_t func_id, void *data,
                                       uint32_t sz);
emis_return_t _emissary_execute_print(uint32_t func_id, void *data,
                                      uint32_t sz);

/// Get uint32 value extended to uint64_t value from a char ptr
uint64_t getuint32(char *val);
/// Get uint64_t value from a char ptr
uint64_t getuint64(char *val);
/// Get a function pointer from a char ptr
void *getfnptr(char *val);
/// build argument array
uint32_t _build_vargs_array(int NumArgs, char *keyptr, char *dataptr,
                            char *strptr, size_t *data_not_used,
                            uint64_t *a[MAXVARGS]);

} // end extern "C"

/// Make the vargs function call to the function pointer fnptr
/// by casting fnptr to vfnptr. Return uint32_t return code
template <typename T, typename FT>
extern uint32_t _call_fnptr(uint32_t NumArgs, void *fnptr,
                            uint64_t *a[MAXVARGS], T *rv);

// Error return codes to _emissary_exec
typedef enum service_rc {
  _RC_SUCCESS = 0,
  _RC_STATUS_UNKNOWN = 1,
  _RC_STATUS_ERROR = 2,
  _RC_STATUS_TERMINATE = 3,
  _RC_DATA_USED_ERROR = 4,
  _RC_ADDINT_ERROR = 5,
  _RC_ADDFLOAT_ERROR = 6,
  _RC_ADDSTRING_ERROR = 7,
  _RC_UNSUPPORTED_ID_ERROR = 8,
  _RC_INVALID_ID_ERROR = 9,
  _RC_ERROR_INVALID_REQUEST = 10,
  _RC_EXCEED_MAXVARGS_ERROR = 11,
  _RC_INVALIDSERVICE_ERROR = 12,
  _RC_ERROR_MEMFREE = 13,
  _RC_ERROR_CONSUMER_ACTIVE = 14,
  _RC_ERROR_CONSUMER_INACTIVE = 15,
  _RC_ERROR_CONSUMER_LAUNCH_FAILED = 16,
  _RC_ERROR_SERVICE_UNKNOWN = 17,
  _RC_ERROR_INCORRECT_ALIGNMENT = 18,
  _RC_ERROR_NULLPTR = 19,
  _RC_ERROR_WRONGVERSION = 20,
  _RC_ERROR_OLDHOSTVERSIONMOD = 21,
  _RC_ERROR_HSAFAIL = 22,
  _RC_ERROR_ZEROPACKETS = 23,
  _RC_ERROR_ALIGNMENT = 24,
} service_rc;

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

#endif // OFFLOAD_EMISSARY_H
