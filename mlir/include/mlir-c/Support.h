//===-- mlir-c/Support.h - Helpers for C API to Core MLIR ---------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the auxiliary data structures used in C APIs to core
// MLIR functionality.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_SUPPORT_H
#define MLIR_C_SUPPORT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// Visibility annotations.
// Use MLIR_CAPI_EXPORTED for exported functions.
//
// On Windows, if MLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC is defined, then
// __declspec(dllexport) and __declspec(dllimport) will be generated. This
// can only be enabled if actually building DLLs. It is generally, mutually
// exclusive with the use of other mechanisms for managing imports/exports
// (i.e. CMake's WINDOWS_EXPORT_ALL_SYMBOLS feature).
//===----------------------------------------------------------------------===//

#if (defined(_WIN32) || defined(__CYGWIN__)) &&                                \
    !defined(MLIR_CAPI_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define MLIR_CAPI_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if MLIR_CAPI_BUILDING_LIBRARY
#define MLIR_CAPI_EXPORTED __declspec(dllexport)
#else
#define MLIR_CAPI_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#define MLIR_CAPI_EXPORTED __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

/// Re-export llvm::ThreadPool so as to avoid including the LLVM C API directly.
DEFINE_C_API_STRUCT(MlirLlvmThreadPool, void);
DEFINE_C_API_STRUCT(MlirTypeID, const void);
DEFINE_C_API_STRUCT(MlirTypeIDAllocator, void);

#undef DEFINE_C_API_STRUCT

//===----------------------------------------------------------------------===//
// MlirStringRef.
//===----------------------------------------------------------------------===//

/// A pointer to a sized fragment of a string, not necessarily null-terminated.
/// Does not own the underlying string. This is equivalent to llvm::StringRef.

struct MlirStringRef {
  const char *data; ///< Pointer to the first symbol.
  size_t length;    ///< Length of the fragment.
};
typedef struct MlirStringRef MlirStringRef;

/// Constructs a string reference from the pointer and length. The pointer need
/// not reference to a null-terminated string.

inline static MlirStringRef mlirStringRefCreate(const char *str,
                                                size_t length) {
  MlirStringRef result;
  result.data = str;
  result.length = length;
  return result;
}

/// Constructs a string reference from a null-terminated C string. Prefer
/// mlirStringRefCreate if the length of the string is known.
MLIR_CAPI_EXPORTED MlirStringRef
mlirStringRefCreateFromCString(const char *str);

/// Returns true if two string references are equal, false otherwise.
MLIR_CAPI_EXPORTED bool mlirStringRefEqual(MlirStringRef string,
                                           MlirStringRef other);

/// A callback for returning string references.
///
/// This function is called back by the functions that need to return a
/// reference to the portion of the string with the following arguments:
///  - an MlirStringRef representing the current portion of the string
///  - a pointer to user data forwarded from the printing call.
typedef void (*MlirStringCallback)(MlirStringRef, void *);

//===----------------------------------------------------------------------===//
// MlirLogicalResult.
//===----------------------------------------------------------------------===//

/// A logical result value, essentially a boolean with named states. LLVM
/// convention for using boolean values to designate success or failure of an
/// operation is a moving target, so MLIR opted for an explicit class.
/// Instances of MlirLogicalResult must only be inspected using the associated
/// functions.
struct MlirLogicalResult {
  int8_t value;
};
typedef struct MlirLogicalResult MlirLogicalResult;

/// Checks if the given logical result represents a success.
inline static bool mlirLogicalResultIsSuccess(MlirLogicalResult res) {
  return res.value != 0;
}

/// Checks if the given logical result represents a failure.
inline static bool mlirLogicalResultIsFailure(MlirLogicalResult res) {
  return res.value == 0;
}

/// Creates a logical result representing a success.
inline static MlirLogicalResult mlirLogicalResultSuccess(void) {
  MlirLogicalResult res = {1};
  return res;
}

/// Creates a logical result representing a failure.
inline static MlirLogicalResult mlirLogicalResultFailure(void) {
  MlirLogicalResult res = {0};
  return res;
}

//===----------------------------------------------------------------------===//
// MlirLlvmThreadPool.
//===----------------------------------------------------------------------===//

/// Create an LLVM thread pool. This is reexported here to avoid directly
/// pulling in the LLVM headers directly.
MLIR_CAPI_EXPORTED MlirLlvmThreadPool mlirLlvmThreadPoolCreate(void);

/// Destroy an LLVM thread pool.
MLIR_CAPI_EXPORTED void mlirLlvmThreadPoolDestroy(MlirLlvmThreadPool pool);

//===----------------------------------------------------------------------===//
// TypeID API.
//===----------------------------------------------------------------------===//

/// `ptr` must be 8 byte aligned and unique to a type valid for the duration of
/// the returned type id's usage
MLIR_CAPI_EXPORTED MlirTypeID mlirTypeIDCreate(const void *ptr);

/// Checks whether a type id is null.
static inline bool mlirTypeIDIsNull(MlirTypeID typeID) { return !typeID.ptr; }

/// Checks if two type ids are equal.
MLIR_CAPI_EXPORTED bool mlirTypeIDEqual(MlirTypeID typeID1, MlirTypeID typeID2);

/// Returns the hash value of the type id.
MLIR_CAPI_EXPORTED size_t mlirTypeIDHashValue(MlirTypeID typeID);

//===----------------------------------------------------------------------===//
// TypeIDAllocator API.
//===----------------------------------------------------------------------===//

/// Creates a type id allocator for dynamic type id creation
MLIR_CAPI_EXPORTED MlirTypeIDAllocator mlirTypeIDAllocatorCreate(void);

/// Deallocates the allocator and all allocated type ids
MLIR_CAPI_EXPORTED void
mlirTypeIDAllocatorDestroy(MlirTypeIDAllocator allocator);

/// Allocates a type id that is valid for the lifetime of the allocator
MLIR_CAPI_EXPORTED MlirTypeID
mlirTypeIDAllocatorAllocateTypeID(MlirTypeIDAllocator allocator);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_SUPPORT_H
