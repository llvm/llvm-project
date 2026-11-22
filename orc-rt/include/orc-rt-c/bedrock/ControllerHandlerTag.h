/*===-- ControllerHandlerTag.h - Controller handler tags ----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Defines the tag type used to identify handlers in the controller.          *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_BEDROCK_CONTROLLERHANDLERTAG_H
#define ORC_RT_C_BEDROCK_CONTROLLERHANDLERTAG_H

#include "orc-rt-c/support/Compiler.h"

ORC_RT_C_EXTERN_C_BEGIN

/**
 * A tag whose address identifies a handler in the controller that is callable
 * by the executor.
 *
 * Tags are global variables that the controller finds by name. Only the
 * address of a tag's definition is meaningful: its contents are never read.
 *
 * Tags must be defined with ORC_RT_C_DEFINE_CONTROLLER_HANDLER_TAG. APIs take
 * tags by non-const pointer (see ORC_RT_C_DEFINE_CONTROLLER_HANDLER_TAG).
 */
typedef struct orc_rt_ControllerHandlerTag {
  char Reserved; /* Non-empty so that every tag has a distinct address. */
} orc_rt_ControllerHandlerTag;

ORC_RT_C_EXTERN_C_END

/**
 * Define a controller handler tag with the given name.
 *
 * The definition has C linkage so that the controller can find it by name.
 * It is explicitly initialized: in C++ this is what makes it a definition, and
 * in C it ensures duplicate tag names are diagnosed even under -fcommon.
 *
 * Tags are deliberately non-const: linkers may fold identical read-only data,
 * which would give distinct tags the same address. APIs take tags by non-const
 * pointer so that passing the address of a const tag is diagnosed.
 */
#ifdef __cplusplus
#define ORC_RT_C_DEFINE_CONTROLLER_HANDLER_TAG(Name)                           \
  extern "C" ORC_RT_C_EXPORT orc_rt_ControllerHandlerTag Name = {0}
#else
#define ORC_RT_C_DEFINE_CONTROLLER_HANDLER_TAG(Name)                           \
  ORC_RT_C_EXPORT orc_rt_ControllerHandlerTag Name = {0}
#endif

#endif /* ORC_RT_C_BEDROCK_CONTROLLERHANDLERTAG_H */
