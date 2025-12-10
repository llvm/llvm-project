/*===-- CoreTypes.h - Essential types for the ORC Runtime C APIs --*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Defines core types for the ORC runtime.                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_CORETYPES_H
#define ORC_RT_C_CORETYPES_H

#include "orc-rt-c/ExternC.h"

ORC_RT_C_EXTERN_C_BEGIN

/**
 * A reference to an orc_rt::Session instance.
 */
typedef struct orc_rt_OpaqueSession *orc_rt_SessionRef;

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_C_CORETYPES_H */
