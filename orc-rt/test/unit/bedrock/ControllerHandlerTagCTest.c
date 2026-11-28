/*===- ControllerHandlerTagCTest.c - Define a tag from C ----------*- C -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Defines a controller handler tag from C, for ControllerHandlerTagTest.cpp. *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "orc-rt-c/bedrock/ControllerHandlerTag.h"

ORC_RT_C_DEFINE_CONTROLLER_HANDLER_TAG(orc_rt_tag_test_controllerHandlerTagC);

/* References the tag, so the test binary fails to link if the macro only
   declared it. Called from ControllerHandlerTagTest.cpp so that the reference
   can't be dead-stripped. */
orc_rt_ControllerHandlerTag *orc_rt_test_controllerHandlerTagC(void) {
  return &orc_rt_tag_test_controllerHandlerTagC;
}
