/*===------------- RTTI.h - C API for ORC Runtime RTTI ------------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines the C interface to the ORC runtime's RTTI functions      *|
|*                                                                            *|
|* TODO: Explain ownership model.                                             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_SUPPORT_RTTI_H
#define ORC_RT_C_SUPPORT_RTTI_H

#include "orc-rt-c/support/Compiler.h"
#include "orc-rt-c/support/CoreTypes.h"

ORC_RT_C_EXTERN_C_BEGIN

/**
 * Opaque reference to an RTTIRoot instance.
 */
typedef struct orc_rt_OpaqueRTTIRoot *orc_rt_RTTIRootRef;

/**
 * Mark a given type as participating in the ORC runtime's RTTI hierarchy.
 *
 * This enables the ORC_RT_DYNCAST operation to be used to safely cast between
 * types.
 */
#define ORC_RT_RTTI_PARTICIPANT(Type)                                          \
  ORC_RT_C_EXPORT orc_rt_##Type##Ref orc_rt_##Type##_fromRTTIRoot(             \
      orc_rt_RTTIRootRef Obj) ORC_RT_C_NOTHROW;                                \
  ORC_RT_MAYBE_UNUSED static inline orc_rt_RTTIRootRef                         \
  orc_rt_##Type##_toRTTIRoot(orc_rt_##Type##Ref Obj) {                         \
    return (orc_rt_RTTIRootRef)Obj;                                            \
  }

#define ORC_RT_DYNCAST(ToType, FromType, Value)                                \
  orc_rt_##ToType##_fromRTTIRoot(orc_rt_##FromType##_toRTTIRoot(Value))

/**
 * Returns the dynamic type name of the given object.
 *
 * For logging purposes only. Use ORC_RT_DYNCAST to test/convert types.
 */
ORC_RT_C_EXPORT const char *
orc_rt_RTTIRoot_getTypeName(orc_rt_RTTIRootRef Obj) ORC_RT_C_NOTHROW;

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_C_SUPPORT_RTTI_H */
