/*===----------- Session.h - ORC Runtime Session C APIs -----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* ORC Runtime Session C APIs.                                                *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ORC_RT_C_BEDROCK_SESSION_H
#define ORC_RT_C_BEDROCK_SESSION_H

#include "orc-rt-c/support/Compiler.h"
#include "orc-rt-c/support/CoreTypes.h"
#include "orc-rt-c/support/WrapperFunction.h"

ORC_RT_C_EXTERN_C_BEGIN

typedef void (*orc_rt_Session_CallControllerReturn)(
    orc_rt_SessionRef S, orc_rt_WrapperFunctionBuffer ResultBytes, void *Ctx);

void orc_rt_Session_callController(orc_rt_SessionRef S,
                                   orc_rt_ControllerHandlerTag T,
                                   orc_rt_WrapperFunctionBuffer ArgBytes,
                                   orc_rt_Session_CallControllerReturn Return,
                                   void *ReturnCtx);

ORC_RT_C_EXTERN_C_END

#endif /* ORC_RT_C_BEDROCK_SESSION_H */
